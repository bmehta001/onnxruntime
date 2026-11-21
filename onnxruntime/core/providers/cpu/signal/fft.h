// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Mixed-radix FFT used by the CPU DFT and STFT kernels.
//
// Lengths whose prime factors are small are decomposed into radix-4/2/3/5 and generic odd-radix passes of a
// self-sorting (Stockham) Cooley-Tukey FFT. Lengths with a large prime factor use Bluestein's chirp-z algorithm on
// top of a smooth-length FFT, so every length runs in O(n log n). Twiddle factors are computed in double precision
// once per plan and reused by every transform executed with the plan. Plans are immutable after construction and
// may be shared by concurrent callers; each caller provides its own scratch memory.

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numbers>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_M_ARM64) || defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define ORT_FFT_FLOAT32X4_NEON 1
#elif (defined(_M_X64) && !defined(_M_ARM64EC)) || defined(__x86_64__) || defined(__SSE2__) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define ORT_FFT_FLOAT32X4_SSE2 1
#endif

namespace onnxruntime {
namespace signal {
namespace fft {

template <typename T>
struct Complex {
  T re;
  T im;
};

static_assert(sizeof(Complex<float>) == sizeof(std::complex<float>));
static_assert(sizeof(Complex<double>) == sizeof(std::complex<double>));

template <typename T>
inline Complex<T> operator+(Complex<T> a, Complex<T> b) { return {a.re + b.re, a.im + b.im}; }

template <typename T>
inline Complex<T> operator-(Complex<T> a, Complex<T> b) { return {a.re - b.re, a.im - b.im}; }

// Four float lanes processed with SSE2 or NEON. Operations map one-to-one to the scalar operators.
#if defined(ORT_FFT_FLOAT32X4_NEON)
#define ORT_FFT_FLOAT32X4 1
struct Float32x4 {
  float32x4_t v;
};
inline Float32x4 operator+(Float32x4 a, Float32x4 b) { return {vaddq_f32(a.v, b.v)}; }
inline Float32x4 operator-(Float32x4 a, Float32x4 b) { return {vsubq_f32(a.v, b.v)}; }
inline Float32x4 operator*(Float32x4 a, Float32x4 b) { return {vmulq_f32(a.v, b.v)}; }
inline Float32x4 operator*(Float32x4 a, float b) { return {vmulq_n_f32(a.v, b)}; }
inline Float32x4 operator-(Float32x4 a) { return {vnegq_f32(a.v)}; }
#elif defined(ORT_FFT_FLOAT32X4_SSE2)
#define ORT_FFT_FLOAT32X4 1
struct Float32x4 {
  __m128 v;
};
inline Float32x4 operator+(Float32x4 a, Float32x4 b) { return {_mm_add_ps(a.v, b.v)}; }
inline Float32x4 operator-(Float32x4 a, Float32x4 b) { return {_mm_sub_ps(a.v, b.v)}; }
inline Float32x4 operator*(Float32x4 a, Float32x4 b) { return {_mm_mul_ps(a.v, b.v)}; }
inline Float32x4 operator*(Float32x4 a, float b) { return {_mm_mul_ps(a.v, _mm_set1_ps(b))}; }
inline Float32x4 operator-(Float32x4 a) { return {_mm_xor_ps(a.v, _mm_set1_ps(-0.0f))}; }
#endif

// Number of independent signals transformed together by a lane type, and the scalar type of each lane.
template <typename V>
struct LaneTraits {
  static constexpr size_t kLanes = 1;
  using Scalar = V;
};

#if defined(ORT_FFT_FLOAT32X4)
template <>
struct LaneTraits<Float32x4> {
  static constexpr size_t kLanes = 4;
  using Scalar = float;
};
#endif

// A Complex<V> occupies 2 * kLanes scalars: the real parts of every lane followed by the imaginary parts.
template <typename V>
inline typename LaneTraits<V>::Scalar* LanePtr(Complex<V>* c) {
  return reinterpret_cast<typename LaneTraits<V>::Scalar*>(c);
}

// The transforms below are written for a "lane" type V, which is either the scalar T or a SIMD vector holding the
// same element of several independent signals (one signal per lane). Plan constants are always scalar T and are
// broadcast across lanes, so every lane performs exactly the operations of the scalar code.

template <typename V, typename S>
inline Complex<V> Scale(Complex<V> a, S s) { return {a.re * s, a.im * s}; }

template <typename V, typename W>
inline Complex<V> Mul(Complex<V> a, Complex<W> b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

template <typename V, typename W>
inline Complex<V> MulConj(Complex<V> a, Complex<W> b) {
  return {a.re * b.re + a.im * b.im, a.im * b.re - a.re * b.im};
}

template <typename T>
inline Complex<T> Conj(Complex<T> a) { return {a.re, -a.im}; }

// Twiddles are stored for the forward direction (exp(-i*theta)); the backward direction uses their conjugates.
template <bool kForward, typename V, typename T>
inline Complex<V> Twiddle(Complex<V> a, Complex<T> w) {
  if constexpr (kForward) {
    return Mul(a, w);
  } else {
    return MulConj(a, w);
  }
}

// Multiplies by -i in the forward direction and by +i in the backward direction.
template <bool kForward, typename T>
inline Complex<T> RotateQuarter(Complex<T> a) {
  if constexpr (kForward) {
    return {a.im, -a.re};
  } else {
    return {-a.im, a.re};
  }
}

// exp(-2*pi*i*k/n) for k in [0, n), computed in double precision.
inline std::vector<Complex<double>> ComputeRoots(size_t n) {
  std::vector<Complex<double>> roots(n);
  constexpr double kTwoPi = 2.0 * std::numbers::pi;
  const size_t half = n / 2;
  for (size_t k = 0; k <= half && k < n; ++k) {
    const double angle = kTwoPi * static_cast<double>(k) / static_cast<double>(n);
    roots[k] = {std::cos(angle), -std::sin(angle)};
  }
  for (size_t k = half + 1; k < n; ++k) {
    roots[k] = Conj(roots[n - k]);
  }
  return roots;
}

// Smallest 2^a * 3^b * 5^c that is >= n.
inline size_t NextSmoothSize(size_t n) {
  if (n <= 1) {
    return 1;
  }
  size_t best = 1;
  while (best < n) {
    best <<= 1;
  }
  for (size_t f5 = 1; f5 < best; f5 *= 5) {
    for (size_t f35 = f5; f35 < best; f35 *= 3) {
      size_t candidate = f35;
      while (candidate < n) {
        candidate <<= 1;
      }
      best = std::min(best, candidate);
    }
  }
  return best;
}

inline std::vector<size_t> Factorize(size_t n) {
  std::vector<size_t> factors;
  while (n % 4 == 0) {
    factors.push_back(4);
    n /= 4;
  }
  if (n % 2 == 0) {
    factors.push_back(2);
    n /= 2;
  }
  for (size_t f = 3; f * f <= n; f += 2) {
    while (n % f == 0) {
      factors.push_back(f);
      n /= f;
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  return factors;
}

// Rough operation count of the direct mixed-radix algorithm; generic odd radices are O(p) per element.
inline double EstimateDirectCost(size_t n) {
  double sum = 0;
  for (size_t f : Factorize(n)) {
    sum += (f <= 5) ? static_cast<double>(f) : 1.1 * static_cast<double>(f);
  }
  return static_cast<double>(n) * sum;
}

template <typename T>
class ComplexFftPlan {
 public:
  explicit ComplexFftPlan(size_t n) : n_(n) {
    if (n_ <= 1) {
      return;
    }
    const size_t m = NextSmoothSize(2 * n_ - 1);
    // Bluestein costs two length-m FFTs plus O(m) pointwise work.
    use_bluestein_ = n_ > 64 && 2.5 * EstimateDirectCost(m) < EstimateDirectCost(n_);
    if (use_bluestein_) {
      InitBluestein(m);
    } else {
      InitDirect();
    }
  }

  size_t Size() const { return n_; }

  // Number of Complex<T> elements of scratch memory required by Forward/Backward.
  size_t ScratchSize() const {
    if (use_bluestein_) {
      return bluestein_m_ + inner_->ScratchSize();
    }
    return n_;
  }

  // Unnormalized in-place transforms of `data` (n elements).
  // V is T or a lane type whose LaneTraits<V>::Scalar is T, transforming one signal per lane.
  template <typename V = T>
  void Forward(Complex<V>* data, Complex<V>* scratch) const { Execute<true>(data, scratch); }
  template <typename V = T>
  void Backward(Complex<V>* data, Complex<V>* scratch) const { Execute<false>(data, scratch); }

 private:
  struct Pass {
    size_t radix;
    size_t l1;
    size_t ido;
    size_t twiddle_offset;
    size_t roots_offset;
  };

  void InitDirect() {
    const auto roots = ComputeRoots(n_);
    size_t l1 = 1;
    for (size_t radix : Factorize(n_)) {
      const size_t ido = n_ / (l1 * radix);
      Pass pass{radix, l1, ido, twiddles_.size(), radix_roots_.size()};
      for (size_t j = 1; j < radix; ++j) {
        for (size_t i = 1; i < ido; ++i) {
          // j * l1 * i < radix * l1 * ido == n, so no modular reduction is needed.
          const auto& w = roots[j * l1 * i];
          twiddles_.push_back({static_cast<T>(w.re), static_cast<T>(w.im)});
        }
      }
      if (radix > 5) {
        const size_t stride = n_ / radix;
        for (size_t j = 0; j < radix; ++j) {
          const auto& w = roots[j * stride];
          radix_roots_.push_back({static_cast<T>(w.re), static_cast<T>(w.im)});
        }
      }
      passes_.push_back(pass);
      l1 *= radix;
    }
  }

  void InitBluestein(size_t m) {
    bluestein_m_ = m;
    inner_ = std::make_unique<ComplexFftPlan<T>>(m);
    std::vector<Complex<double>> chirp(n_);
    const uint64_t two_n = 2 * static_cast<uint64_t>(n_);
    for (size_t k = 0; k < n_; ++k) {
      // exp(-i*pi*k^2/n); reduce k^2 modulo 2n exactly so large k keep full precision.
      const uint64_t k2 = (static_cast<uint64_t>(k) * k) % two_n;
      const double angle = std::numbers::pi * static_cast<double>(k2) / static_cast<double>(n_);
      chirp[k] = {std::cos(angle), -std::sin(angle)};
    }
    std::vector<Complex<T>> b(m, Complex<T>{0, 0});
    const double inv_m = 1.0 / static_cast<double>(m);
    for (size_t k = 0; k < n_; ++k) {
      const Complex<T> v{static_cast<T>(chirp[k].re * inv_m), static_cast<T>(-chirp[k].im * inv_m)};
      b[k] = v;
      if (k != 0) {
        b[m - k] = v;
      }
    }
    std::vector<Complex<T>> scratch(inner_->ScratchSize());
    inner_->Forward(b.data(), scratch.data());
    kernel_fft_ = std::move(b);
    chirp_.reserve(n_);
    for (const auto& w : chirp) {
      chirp_.push_back({static_cast<T>(w.re), static_cast<T>(w.im)});
    }
  }

  template <bool kForward, typename V>
  void Execute(Complex<V>* data, Complex<V>* scratch) const {
    if (n_ <= 1) {
      return;
    }
    if (use_bluestein_) {
      ExecuteBluestein<kForward>(data, scratch);
      return;
    }
    Complex<V>* in = data;
    Complex<V>* out = scratch;
    for (const auto& pass : passes_) {
      const Complex<T>* tw = twiddles_.data() + pass.twiddle_offset;
      switch (pass.radix) {
        case 4:
          Pass4<kForward>(pass.ido, pass.l1, in, out, tw);
          break;
        case 2:
          Pass2<kForward>(pass.ido, pass.l1, in, out, tw);
          break;
        case 3:
          Pass3<kForward>(pass.ido, pass.l1, in, out, tw);
          break;
        case 5:
          Pass5<kForward>(pass.ido, pass.l1, in, out, tw);
          break;
        case 7:
          PassOdd<kForward, 7>(7, pass.ido, pass.l1, in, out, tw, radix_roots_.data() + pass.roots_offset);
          break;
        case 11:
          PassOdd<kForward, 11>(11, pass.ido, pass.l1, in, out, tw, radix_roots_.data() + pass.roots_offset);
          break;
        default:
          PassOdd<kForward, 0>(pass.radix, pass.ido, pass.l1, in, out, tw, radix_roots_.data() + pass.roots_offset);
          break;
      }
      std::swap(in, out);
    }
    if (in != data) {
      std::copy(in, in + n_, data);
    }
  }

  template <bool kForward, typename V>
  void ExecuteBluestein(Complex<V>* data, Complex<V>* scratch) const {
    const size_t m = bluestein_m_;
    Complex<V>* a = scratch;
    Complex<V>* inner_scratch = scratch + m;
    // The backward transform is conj(forward(conj(x))).
    for (size_t k = 0; k < n_; ++k) {
      Complex<V> x = kForward ? data[k] : Conj(data[k]);
      a[k] = Mul(x, chirp_[k]);
    }
    std::fill(a + n_, a + m, Complex<V>{});
    inner_->Forward(a, inner_scratch);
    for (size_t k = 0; k < m; ++k) {
      a[k] = Mul(a[k], kernel_fft_[k]);
    }
    inner_->Backward(a, inner_scratch);
    for (size_t k = 0; k < n_; ++k) {
      Complex<V> y = Mul(a[k], chirp_[k]);
      data[k] = kForward ? y : Conj(y);
    }
  }

  // Passes follow the self-sorting decimation-in-frequency layout: input element (i, j, k) is at
  // in[i + ido * (j + radix * k)] and output element (i, k, j) is written to out[i + ido * (k + l1 * j)], with the
  // output twiddle exp(-2*pi*i * j*l1*i / n) applied to every non-zero j and i.

  template <bool kForward, typename V>
  static void Pass2(size_t ido, size_t l1, const Complex<V>* in, Complex<V>* out, const Complex<T>* tw) {
    for (size_t k = 0; k < l1; ++k) {
      const Complex<V>* src = in + ido * 2 * k;
      Complex<V>* dst0 = out + ido * k;
      Complex<V>* dst1 = out + ido * (k + l1);
      dst0[0] = src[0] + src[ido];
      dst1[0] = src[0] - src[ido];
      for (size_t i = 1; i < ido; ++i) {
        const Complex<V> a = src[i];
        const Complex<V> b = src[i + ido];
        dst0[i] = a + b;
        dst1[i] = Twiddle<kForward>(a - b, tw[i - 1]);
      }
    }
  }

  template <bool kForward, typename V>
  static void Pass3(size_t ido, size_t l1, const Complex<V>* in, Complex<V>* out, const Complex<T>* tw) {
    constexpr T kHalf = static_cast<T>(0.5);
    constexpr T kSin60 = static_cast<T>(0.86602540378443864676372317075294);
    const Complex<T>* tw1 = tw;
    const Complex<T>* tw2 = tw + (ido - 1);
    for (size_t k = 0; k < l1; ++k) {
      const Complex<V>* src = in + ido * 3 * k;
      Complex<V>* dst0 = out + ido * k;
      Complex<V>* dst1 = out + ido * (k + l1);
      Complex<V>* dst2 = out + ido * (k + 2 * l1);
      for (size_t i = 0; i < ido; ++i) {
        const Complex<V> x0 = src[i];
        const Complex<V> x1 = src[i + ido];
        const Complex<V> x2 = src[i + 2 * ido];
        const Complex<V> t1 = x1 + x2;
        const Complex<V> t2 = Scale(x1 - x2, kSin60);
        const Complex<V> ca = x0 - Scale(t1, kHalf);
        const Complex<V> cb = RotateQuarter<kForward>(t2);
        dst0[i] = x0 + t1;
        if (i == 0) {
          dst1[i] = ca + cb;
          dst2[i] = ca - cb;
        } else {
          dst1[i] = Twiddle<kForward>(ca + cb, tw1[i - 1]);
          dst2[i] = Twiddle<kForward>(ca - cb, tw2[i - 1]);
        }
      }
    }
  }

  template <bool kForward, typename V>
  static void Pass4(size_t ido, size_t l1, const Complex<V>* in, Complex<V>* out, const Complex<T>* tw) {
    const Complex<T>* tw1 = tw;
    const Complex<T>* tw2 = tw + (ido - 1);
    const Complex<T>* tw3 = tw + 2 * (ido - 1);
    for (size_t k = 0; k < l1; ++k) {
      const Complex<V>* src = in + ido * 4 * k;
      Complex<V>* dst0 = out + ido * k;
      Complex<V>* dst1 = out + ido * (k + l1);
      Complex<V>* dst2 = out + ido * (k + 2 * l1);
      Complex<V>* dst3 = out + ido * (k + 3 * l1);
      {
        const Complex<V> t0 = src[0] + src[2 * ido];
        const Complex<V> t1 = src[0] - src[2 * ido];
        const Complex<V> t2 = src[ido] + src[3 * ido];
        const Complex<V> t3 = RotateQuarter<kForward>(src[ido] - src[3 * ido]);
        dst0[0] = t0 + t2;
        dst1[0] = t1 + t3;
        dst2[0] = t0 - t2;
        dst3[0] = t1 - t3;
      }
      for (size_t i = 1; i < ido; ++i) {
        const Complex<V> x0 = src[i];
        const Complex<V> x1 = src[i + ido];
        const Complex<V> x2 = src[i + 2 * ido];
        const Complex<V> x3 = src[i + 3 * ido];
        const Complex<V> t0 = x0 + x2;
        const Complex<V> t1 = x0 - x2;
        const Complex<V> t2 = x1 + x3;
        const Complex<V> t3 = RotateQuarter<kForward>(x1 - x3);
        dst0[i] = t0 + t2;
        dst1[i] = Twiddle<kForward>(t1 + t3, tw1[i - 1]);
        dst2[i] = Twiddle<kForward>(t0 - t2, tw2[i - 1]);
        dst3[i] = Twiddle<kForward>(t1 - t3, tw3[i - 1]);
      }
    }
  }

  template <bool kForward, typename V>
  static void Pass5(size_t ido, size_t l1, const Complex<V>* in, Complex<V>* out, const Complex<T>* tw) {
    constexpr T kC1 = static_cast<T>(0.30901699437494742410229341718282);   // cos(2*pi/5)
    constexpr T kC2 = static_cast<T>(-0.80901699437494742410229341718282);  // cos(4*pi/5)
    constexpr T kS1 = static_cast<T>(0.95105651629515357211643933337938);   // sin(2*pi/5)
    constexpr T kS2 = static_cast<T>(0.58778525229247312916870595463907);   // sin(4*pi/5)
    for (size_t k = 0; k < l1; ++k) {
      const Complex<V>* src = in + ido * 5 * k;
      Complex<V>* dst[5];
      for (size_t j = 0; j < 5; ++j) {
        dst[j] = out + ido * (k + j * l1);
      }
      for (size_t i = 0; i < ido; ++i) {
        const Complex<V> x0 = src[i];
        const Complex<V> x1 = src[i + ido];
        const Complex<V> x2 = src[i + 2 * ido];
        const Complex<V> x3 = src[i + 3 * ido];
        const Complex<V> x4 = src[i + 4 * ido];
        const Complex<V> t1 = x1 + x4;
        const Complex<V> t4 = x1 - x4;
        const Complex<V> t2 = x2 + x3;
        const Complex<V> t3 = x2 - x3;
        const Complex<V> a1 = x0 + Scale(t1, kC1) + Scale(t2, kC2);
        const Complex<V> a2 = x0 + Scale(t1, kC2) + Scale(t2, kC1);
        const Complex<V> b1 = RotateQuarter<kForward>(Scale(t4, kS1) + Scale(t3, kS2));
        const Complex<V> b2 = RotateQuarter<kForward>(Scale(t4, kS2) - Scale(t3, kS1));
        dst[0][i] = x0 + t1 + t2;
        if (i == 0) {
          dst[1][i] = a1 + b1;
          dst[4][i] = a1 - b1;
          dst[2][i] = a2 + b2;
          dst[3][i] = a2 - b2;
        } else {
          dst[1][i] = Twiddle<kForward>(a1 + b1, tw[i - 1]);
          dst[2][i] = Twiddle<kForward>(a2 + b2, tw[(ido - 1) + i - 1]);
          dst[3][i] = Twiddle<kForward>(a2 - b2, tw[2 * (ido - 1) + i - 1]);
          dst[4][i] = Twiddle<kForward>(a1 - b1, tw[3 * (ido - 1) + i - 1]);
        }
      }
    }
  }

  // Odd radix p > 5. `roots` holds exp(-2*pi*i*j/p) for j in [0, p). A non-zero kRadix fixes p at compile time so
  // the butterfly is fully unrolled.
  template <bool kForward, size_t kRadix, typename V>
  static void PassOdd(size_t runtime_radix, size_t ido, size_t l1, const Complex<V>* in, Complex<V>* out,
                      const Complex<T>* tw, const Complex<T>* roots) {
    const size_t radix = kRadix != 0 ? kRadix : runtime_radix;
    const size_t half = radix / 2;
    for (size_t k = 0; k < l1; ++k) {
      const Complex<V>* src = in + ido * radix * k;
      for (size_t i = 0; i < ido; ++i) {
        const Complex<V> x0 = src[i];
        Complex<V> dc = x0;
        for (size_t j = 1; j <= half; ++j) {
          dc = dc + src[i + j * ido] + src[i + (radix - j) * ido];
        }
        out[i + ido * k] = dc;
        for (size_t m = 1; m <= half; ++m) {
          // y[m] = x0 + sum_j cos(2*pi*j*m/p) * (x[j] + x[p-j]) + rot(sum_j sin(2*pi*j*m/p) * (x[j] - x[p-j])).
          Complex<V> sum_cos = x0;
          Complex<V> sum_sin{};
          size_t jm = 0;
          for (size_t j = 1; j <= half; ++j) {
            jm += m;
            if (jm >= radix) {
              jm -= radix;
            }
            const Complex<V> xp = src[i + j * ido];
            const Complex<V> xm = src[i + (radix - j) * ido];
            const Complex<T> w = roots[jm];
            sum_cos = sum_cos + Scale(xp + xm, w.re);
            // roots[] store -sin in the imaginary part.
            sum_sin = sum_sin + Scale(xp - xm, -w.im);
          }
          const Complex<V> rot = RotateQuarter<kForward>(sum_sin);
          Complex<V> y_m = sum_cos + rot;
          Complex<V> y_pm = sum_cos - rot;
          if (i != 0) {
            y_m = Twiddle<kForward>(y_m, tw[(m - 1) * (ido - 1) + i - 1]);
            y_pm = Twiddle<kForward>(y_pm, tw[(radix - m - 1) * (ido - 1) + i - 1]);
          }
          out[i + ido * (k + m * l1)] = y_m;
          out[i + ido * (k + (radix - m) * l1)] = y_pm;
        }
      }
    }
  }

  size_t n_;
  bool use_bluestein_ = false;
  std::vector<Pass> passes_;
  std::vector<Complex<T>> twiddles_;
  std::vector<Complex<T>> radix_roots_;
  size_t bluestein_m_ = 0;
  std::unique_ptr<ComplexFftPlan<T>> inner_;
  std::vector<Complex<T>> chirp_;
  std::vector<Complex<T>> kernel_fft_;
};

// FFT of a real signal of length n. For even n the signal is packed into an n/2-point complex FFT followed by a
// split step; odd n falls back to a full complex FFT.
template <typename T>
class RealFftPlan {
 public:
  explicit RealFftPlan(size_t n) : n_(n), half_(n / 2) {
    if (n_ % 2 == 0 && n_ >= 2) {
      complex_plan_ = std::make_unique<ComplexFftPlan<T>>(half_);
      const auto roots = ComputeRoots(n_);
      split_twiddles_.resize(half_ + 1);
      for (size_t k = 0; k <= half_; ++k) {
        const auto& w = roots[k % n_];
        split_twiddles_[k] = {static_cast<T>(w.re), static_cast<T>(w.im)};
      }
    } else {
      complex_plan_ = std::make_unique<ComplexFftPlan<T>>(n_);
    }
  }

  size_t Size() const { return n_; }
  bool IsPacked() const { return n_ % 2 == 0 && n_ >= 2; }

  // Number of Complex<T> elements needed for the in-place buffer passed to Forward/Backward.
  size_t BufferSize() const { return IsPacked() ? half_ + 1 : n_; }
  size_t ScratchSize() const { return complex_plan_->ScratchSize(); }

  // On entry `buffer` holds n real samples: packed as n consecutive T values when IsPacked(), otherwise as n complex
  // values with zero imaginary parts. On exit buffer[0..n/2] holds the non-negative frequency bins.
  template <typename V = T>
  void Forward(Complex<V>* buffer, Complex<V>* scratch) const {
    complex_plan_->Forward(buffer, scratch);
    if (!IsPacked()) {
      return;
    }
    const size_t h = half_;
    const Complex<V> z0 = buffer[0];
    buffer[0] = {z0.re + z0.im, V{}};
    buffer[h] = {z0.re - z0.im, V{}};
    constexpr T kHalf = static_cast<T>(0.5);
    for (size_t k = 1, j = h - 1; k <= j; ++k, --j) {
      const Complex<V> zk = buffer[k];
      const Complex<V> zj = buffer[j];
      // X[k] = E + w^k * O, X[j] = conj(E) + w^j * conj(O) with E = (Z[k] + conj(Z[j])) / 2 and
      // O = (Z[k] - conj(Z[j])) / 2i, where j = h - k.
      const Complex<V> e = Scale(zk + Conj(zj), kHalf);
      const Complex<V> d = Scale(zk - Conj(zj), kHalf);
      const Complex<V> o{d.im, -d.re};
      buffer[k] = e + Mul(o, split_twiddles_[k]);
      if (j != k) {
        buffer[j] = Conj(e) + Mul(Conj(o), split_twiddles_[j]);
      }
    }
  }

  // On entry buffer[0..n/2] holds non-negative frequency bins of a Hermitian spectrum (the imaginary parts of the DC
  // and, for even n, Nyquist bins are ignored). On exit `buffer` holds the n real samples scaled by n, packed as
  // described for Forward. Only valid when IsPacked().
  template <typename V = T>
  void Backward(Complex<V>* buffer, Complex<V>* scratch) const {
    const size_t h = half_;
    const V x0 = buffer[0].re;
    const V xh = buffer[h].re;
    buffer[0] = {x0 + xh, x0 - xh};
    for (size_t k = 1, j = h - 1; k <= j; ++k, --j) {
      const Complex<V> xk = buffer[k];
      const Complex<V> xj = buffer[j];
      // Z[k] = (X[k] + conj(X[j])) + i * (X[k] - conj(X[j])) * conj(w^k), j = h - k.
      const Complex<V> e = xk + Conj(xj);
      const Complex<V> d = MulConj(xk - Conj(xj), split_twiddles_[k]);
      buffer[k] = e + Complex<V>{-d.im, d.re};
      if (j != k) {
        const Complex<V> e2 = xj + Conj(xk);
        const Complex<V> d2 = MulConj(xj - Conj(xk), split_twiddles_[j]);
        buffer[j] = e2 + Complex<V>{-d2.im, d2.re};
      }
    }
    complex_plan_->Backward(buffer, scratch);
  }

  const ComplexFftPlan<T>& ComplexPlan() const { return *complex_plan_; }

 private:
  size_t n_;
  size_t half_;
  std::unique_ptr<ComplexFftPlan<T>> complex_plan_;
  std::vector<Complex<T>> split_twiddles_;
};

// Remembers the most recently used plan of a given kind. Building a plan costs O(n) trigonometric evaluations, which
// is comparable to running a few transforms, so kernels keep plans across Compute calls. Thread-safe.
template <typename Plan>
class PlanSlot {
 public:
  std::shared_ptr<const Plan> Get(size_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (plan_ == nullptr || plan_->Size() != n) {
      plan_ = std::make_shared<const Plan>(n);
    }
    return plan_;
  }

 private:
  std::mutex mutex_;
  std::shared_ptr<const Plan> plan_;
};

class PlanCache {
 public:
  template <typename T>
  std::shared_ptr<const ComplexFftPlan<T>> GetComplex(size_t n) {
    if constexpr (std::is_same_v<T, float>) {
      return complex_float_.Get(n);
    } else {
      return complex_double_.Get(n);
    }
  }

  template <typename T>
  std::shared_ptr<const RealFftPlan<T>> GetReal(size_t n) {
    if constexpr (std::is_same_v<T, float>) {
      return real_float_.Get(n);
    } else {
      return real_double_.Get(n);
    }
  }

 private:
  PlanSlot<ComplexFftPlan<float>> complex_float_;
  PlanSlot<ComplexFftPlan<double>> complex_double_;
  PlanSlot<RealFftPlan<float>> real_float_;
  PlanSlot<RealFftPlan<double>> real_double_;
};

}  // namespace fft
}  // namespace signal
}  // namespace onnxruntime
