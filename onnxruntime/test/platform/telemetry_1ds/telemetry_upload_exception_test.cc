// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

#include "ILogConfiguration.hpp"
#include "LogManagerProvider.hpp"
#include "config/RuntimeConfig_Default.hpp"
#include "pal/PAL.hpp"
#include "system/TelemetrySystemBase.hpp"
#include "tpm/TransmissionPolicyManager.hpp"

namespace {
namespace sdk = Microsoft::Applications::Events;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::_Exit(1);
  }
}

class NoNetworkClient final : public sdk::IHttpClient {
 public:
  sdk::IHttpRequest* CreateRequest() override {
    Check(false, "component regression unexpectedly created an HTTP request");
    return nullptr;
  }
  void SendRequestAsync(sdk::IHttpRequest*, sdk::IHttpResponseCallback*) override {
    Check(false, "component regression unexpectedly attempted an upload");
  }
  void CancelRequestAsync(const std::string&) override {}
};

enum class Failure {
  Standard,
  Nonstandard,
  CleanupStandard,
  CleanupNonstandard,
  Registered,
  Completed,
  CompletedThenThrow,
};

class UploadProbe final : public sdk::TransmissionPolicyManager {
 public:
  UploadProbe(sdk::ITelemetrySystem& system, sdk::ITaskDispatcher& dispatcher, Failure failure)
      : TransmissionPolicyManager(system, dispatcher, nullptr), failure_(failure) {
    m_isPaused = false;
    initiateUpload >> prepare_;
    releaseFailedUpload >> release_;
  }

  void Run(bool immediate) {
    if (immediate) {
      sdk::IncomingEventContext event;
      event.record.latency = sdk::EventLatency_Max;
      handleEventArrived(&event);
    } else {
      uploadAsync(sdk::EventLatency_Normal);
    }

    Check(prepared_ == 1, "upload entry point did not execute preparation");
    if (failure_ == Failure::Registered) {
      Check(uploadCount() == 1 && !context_.expired(), "HTTP-owned upload was retired prematurely");
      eventsUploadAborted(pending_);
      eventsUploadAborted(pending_);
      pending_.reset();
    }
    Check(uploadCount() == 0, "preparation exception stranded an active upload");
    const bool http_owned = failure_ == Failure::Registered || failure_ == Failure::Completed ||
                            failure_ == Failure::CompletedThenThrow;
    Check(released_ == (http_owned ? 0 : 1), "records were released after HTTP ownership or not released on failure");
    Check(context_.expired(), "completed upload context was retained");
    Check(stop(), "transmission manager did not stop");
  }

 private:
  void Prepare(const sdk::EventsUploadContextPtr& ctx) {
    ++prepared_;
    context_ = ctx;
    if (failure_ == Failure::Registered || failure_ == Failure::Completed ||
        failure_ == Failure::CompletedThenThrow) {
      ctx->httpRequestRegistered = true;
      if (failure_ == Failure::Registered) {
        pending_ = ctx;
      } else {
        eventsUploadAborted(ctx);
        if (failure_ == Failure::Completed) {
          return;
        }
      }
    }
    if (failure_ == Failure::Nonstandard) {
      throw 7;
    }
    throw std::runtime_error("injected upload preparation failure");
  }

  void Release(const sdk::EventsUploadContextPtr&) {
    ++released_;
    Check(uploadCount() == 1, "upload was retired before record cleanup finished");
    if (failure_ == Failure::CleanupStandard) {
      throw std::runtime_error("injected record cleanup failure");
    }
    if (failure_ == Failure::CleanupNonstandard) {
      throw 9;
    }
  }

  Failure failure_;
  size_t prepared_ = 0;
  size_t released_ = 0;
  std::weak_ptr<sdk::EventsUploadContext> context_;
  sdk::EventsUploadContextPtr pending_;
  sdk::RouteSink<UploadProbe, const sdk::EventsUploadContextPtr&> prepare_{this, &UploadProbe::Prepare};
  sdk::RouteSink<UploadProbe, const sdk::EventsUploadContextPtr&> release_{this, &UploadProbe::Release};
};
}  // namespace

int main() {
  sdk::ILogConfiguration config;
  config[sdk::CFG_STR_COLLECTOR_URL] = "";
  config[sdk::CFG_STR_CACHE_FILE_PATH] = ":memory:";
  config[sdk::CFG_BOOL_ENABLE_NET_DETECT] = false;
  config[sdk::CFG_INT_MAX_TEARDOWN_TIME] = 0;
  config.AddModule(sdk::CFG_MODULE_HTTP_CLIENT, std::make_shared<NoNetworkClient>());
  sdk::status_t status;
  auto* manager = sdk::LogManagerProvider::CreateLogManager("UploadExceptionTest", true, config, status);
  Check(manager != nullptr && status == sdk::STATUS_SUCCESS, "could not initialize SDK");
  {
    auto dispatcher = sdk::PlatformAbstraction::getDefaultTaskDispatcher();
    Check(dispatcher != nullptr, "SDK worker was unavailable");
    sdk::RuntimeConfig_Default runtime_config(config);
    sdk::TelemetrySystemBase system(*manager, runtime_config, *dispatcher);
    for (bool immediate : {false, true}) {
      for (Failure failure : {Failure::Standard, Failure::Nonstandard, Failure::CleanupStandard,
                              Failure::CleanupNonstandard, Failure::Registered, Failure::Completed,
                              Failure::CompletedThenThrow}) {
        UploadProbe probe(system, *dispatcher, failure);
        probe.Run(immediate);
      }
    }
  }
  manager->FlushAndTeardown();
  Check(sdk::LogManagerProvider::Release(config) == sdk::STATUS_SUCCESS, "could not release SDK");
  std::puts("PASS: 14 scheduled/immediate upload exception and ownership cases");
}
