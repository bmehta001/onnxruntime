// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/skip_layer_norm_fusion.h"
#include "core/graph/graph_utils.h"
#include "float.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// Schema default epsilon of LayerNormalization, SimplifiedLayerNormalization and RMSNormalization.
static constexpr float kNormDefaultEpsilon = 1e-5f;

// LayerNorm supports limited data types.
static constexpr std::array supported_data_types{
    "tensor(float16)", "tensor(float)", "tensor(bfloat16)"};

static bool IsSupportedDataType(const Node& node) {
  for (const auto& input_arg : node.InputDefs()) {
    if (std::find(supported_data_types.begin(), supported_data_types.end(),
                  *(input_arg->Type())) == supported_data_types.end()) {
      return false;
    }
  }
  return true;
}

// The residual sum may feed other nodes (pre-norm transformer blocks) only when it can be re-emitted
// through the fused node's optional input_skip_bias_sum output.
static bool CheckFirstAdd(Node& add, ProviderType providertype, bool allow_sum_output) {
  if (providertype != add.GetExecutionProviderType() ||
      !IsSupportedDataType(add) ||
      add.GetOutputEdgesCount() == 0 ||
      (!allow_sum_output && add.GetOutputEdgesCount() != 1)) {
    return false;
  }

  // Check the input dimensions of the "Add" node.
  const TensorShapeProto* add_input1_shape = add.MutableInputDefs()[0]->Shape();
  const TensorShapeProto* add_input2_shape = add.MutableInputDefs()[1]->Shape();

  if (add_input1_shape == nullptr || add_input2_shape == nullptr) {
    return false;
  }
  // "Add" inputs have to be 3d.
  if (add_input1_shape->dim_size() != 3 || add_input2_shape->dim_size() != 3) {
    return false;
  }
  // "Add" inputs have to be of same dimensions.
  bool is_valid_input = true;
  for (int i = 0; i < 3; i++) {
    if (!utils::HasDimValue(add_input1_shape->dim(i)) ||
        !utils::HasDimValue(add_input2_shape->dim(i)) ||
        add_input1_shape->dim(i).dim_value() != add_input2_shape->dim(i).dim_value()) {
      // Allow dimension only has dim_param.
      if (!utils::HasDimParam(add_input1_shape->dim(i)) ||
          !utils::HasDimParam(add_input2_shape->dim(i)) ||
          add_input1_shape->dim(i).dim_param() != add_input2_shape->dim(i).dim_param()) {
        is_valid_input = false;
        break;
      }
    }
  }
  return is_valid_input;
}

// Add2 is the 2nd add of the to be fused sub-graph
// The 1st input should be a 3D tensor
// The 2nd input should be a 1D constant value
static bool CheckSecondAdd(Graph& graph, Node& add, ProviderType providertype) {
  if (providertype != add.GetExecutionProviderType() ||
      !IsSupportedDataType(add) ||
      add.GetOutputEdgesCount() != 1) {
    return false;
  }

  // The 2nd input should be a constant value
  if (!graph_utils::NodeArgIsConstant(graph, *(add.MutableInputDefs()[1]))) {
    return false;
  }

  // Check the input dimensions of the "Add" node.
  const TensorShapeProto* add_input1_shape = add.MutableInputDefs()[0]->Shape();
  const TensorShapeProto* add_input2_shape = add.MutableInputDefs()[1]->Shape();

  if (add_input1_shape == nullptr || add_input2_shape == nullptr) {
    return false;
  }

  return add_input1_shape->dim_size() == 3 &&
         add_input2_shape->dim_size() == 1 &&
         utils::HasDimValue(add_input1_shape->dim(2)) &&
         utils::HasDimValue(add_input2_shape->dim(0)) &&
         add_input1_shape->dim(2).dim_value() == add_input2_shape->dim(0).dim_value();
}

// Add a Cast to convert input from float16/bfloat16 to float when input type is different fromm output type
static NodeArg* CastToFloat(Graph& graph, NodeArg* input, int32_t output_data_type, ProviderType provider_type) {
  if (nullptr == input->Type() ||
      input->TypeAsProto()->tensor_type().elem_type() == output_data_type ||
      output_data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return input;
  }

  auto input_shape = input->Shape();
  TypeProto input_float;
  input_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  for (auto i = 0; i < input_shape->dim_size(); ++i) {
    auto dim = input_float.mutable_tensor_type()->mutable_shape()->add_dim();
    *dim = input_shape->dim(i);
  }
  auto& cast_float = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(input->Name() + "_Float"), &input_float);

  auto& node = graph.AddNode(graph.GenerateNodeName(input->Name() + "_Cast"),
                             "Cast",
                             "Cast Input to float",
                             std::array{input},
                             std::array{&cast_float},
                             nullptr,
                             kOnnxDomain);

  node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
  node.SetExecutionProviderType(provider_type);
  return &cast_float;
}

static int32_t ElemType(const NodeArg* arg) {
  return arg->TypeAsProto() == nullptr ? ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED
                                       : arg->TypeAsProto()->tensor_type().elem_type();
}

static bool OutputIsUsed(const Graph& graph, const Node& node, size_t output_index) {
  const auto& output_defs = node.OutputDefs();
  if (output_index >= output_defs.size() || !output_defs[output_index]->Exists()) {
    return false;
  }
  if (graph.IsOutput(output_defs[output_index])) {
    return true;
  }
  for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
    if (static_cast<size_t>(it->GetSrcArgIndex()) == output_index) {
      return true;
    }
  }
  return false;
}

// SimplifiedLayerNormalization / RMSNormalization can map onto SkipSimplifiedLayerNormalization only when they
// normalize the last axis in fp32 statistics, use a single tensor type, and only output 0 is consumed (their
// inv_std_var output index does not line up with the fused op's).
static bool IsFusibleSimplifiedNorm(const Graph& graph, const Node& norm) {
  const auto& attrs = norm.GetAttributes();
  if (auto it = attrs.find("axis"); it != attrs.end()) {
    const TensorShapeProto* x_shape = norm.InputDefs()[0]->Shape();
    const int64_t axis = it->second.i();
    if (axis != -1 && (x_shape == nullptr || axis != x_shape->dim_size() - 1)) {
      return false;
    }
  }
  if (auto it = attrs.find("stash_type"); it != attrs.end() &&
                                          it->second.i() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return false;
  }
  const int32_t x_type = ElemType(norm.InputDefs()[0]);
  if (x_type != ElemType(norm.InputDefs()[1]) || x_type != ElemType(norm.OutputDefs()[0])) {
    return false;
  }
  for (size_t i = 1; i < norm.OutputDefs().size(); ++i) {
    if (OutputIsUsed(graph, norm, i)) {
      return false;
    }
  }
  return true;
}

/**
Skip Layer Normalization will fuse Add + LayerNormalization into one node, and another Add if applicable.
On the CPU EP, Add + SimplifiedLayerNormalization (or ONNX RMSNormalization) is likewise fused into
SkipSimplifiedLayerNormalization, and an Add whose sum has other consumers (pre-norm residual) is fused by
re-emitting the sum through the optional input_skip_bias_sum output.

Before fusion:
Format 1:
    [Sub1]  C    [Sub2]
        \  /     /
        Add2    /
           \   /
            Add1
             |
     LayerNormalization

Format 2:
      [Sub1] [Sub2]  C
         \      \   /
          \     Add2
           \    /
            Add1
             |
     LayerNormalization

Format 3:
      [Sub1]   [Sub2]
         \       /
          \     /
           \   /
            Add1
             |
     LayerNormalization

After fusion:
       [Sub1]   [Sub1]
         \      /
          \    /
    SkipLayerNormalization

Note: This fusion doesn't consider the following case:
      [Sub1]   [Sub2]
         \       /
        Add2  Add3
           \   /
            Add1
             |
     LayerNormalization
*/

Status SkipLayerNormFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                      const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  InlinedVector<std::reference_wrapper<Node>> nodes_to_remove;
  InlinedHashSet<NodeIndex> fused_nodes;
  for (auto node_index : node_topology_list) {
    Node* p_layernorm = graph.GetNode(node_index);
    if (p_layernorm == nullptr)
      continue;  // node was removed in an earlier fusion.

    Node& ln_node = *p_layernorm;
    ORT_RETURN_IF_ERROR(Recurse(ln_node, modified, graph_level, logger));

    const bool is_layer_norm = graph_utils::IsSupportedOptypeVersionAndDomain(ln_node, "LayerNormalization", {1, 17});
    const bool is_simplified =
        !is_layer_norm &&
        (graph_utils::IsSupportedOptypeVersionAndDomain(ln_node, "SimplifiedLayerNormalization", {1}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(ln_node, "RMSNormalization", {23}));
    if ((!is_layer_norm && !is_simplified) ||
        !graph_utils::IsSupportedProvider(ln_node, GetCompatibleExecutionProviders()) ||
        !IsSupportedDataType(ln_node)) {
      continue;
    }

    // The simplified-norm and residual-sum-output variants are only validated for the CPU kernels.
    const bool is_cpu_ep = ln_node.GetExecutionProviderType() == kCpuExecutionProvider;
    if (is_simplified && (!is_cpu_ep || !IsFusibleSimplifiedNorm(graph, ln_node))) {
      continue;
    }
    const bool allow_sum_output = is_cpu_ep;

    enum class Format : int8_t {
      Format1,
      Format2,
      Format3,
      None
    };

    Node* p_add1 = nullptr;
    Node* p_add2 = nullptr;
    Format matched_format = Format::None;

    // The matcher paths are passed inline via FindPath's std::initializer_list overload. Their
    // backing arrays are stack temporaries, so they stay clear of the GCC 15 -Wfree-nonheap-object
    // false positive that fired on the equivalent local std::vector's heap-buffer destructor.
    std::vector<const Node::EdgeEnd*> edges;
    // Format 1
    if (graph_utils::FindPath(ln_node, true,
                              {{0, 0, "Add", {7, 13, 14}, kOnnxDomain},
                               {0, 0, "Add", {7, 13, 14}, kOnnxDomain}},
                              edges, logger)) {
      p_add1 = const_cast<Node*>(&edges[0]->GetNode());
      p_add2 = const_cast<Node*>(&edges[1]->GetNode());

      if (CheckFirstAdd(*p_add1, ln_node.GetExecutionProviderType(), allow_sum_output) &&
          CheckSecondAdd(graph, *p_add2, ln_node.GetExecutionProviderType()) &&
          !graph.NodeProducesGraphOutput(*p_add1) &&
          !graph.NodeProducesGraphOutput(*p_add2)) {
        matched_format = Format::Format1;
      }
    }

    if (matched_format == Format::None) {
      // Format 2
      if (graph_utils::FindPath(ln_node, true,
                                {{0, 0, "Add", {7, 13, 14}, kOnnxDomain},
                                 {0, 1, "Add", {7, 13, 14}, kOnnxDomain}},
                                edges, logger)) {
        p_add1 = const_cast<Node*>(&edges[0]->GetNode());
        p_add2 = const_cast<Node*>(&edges[1]->GetNode());

        if (CheckFirstAdd(*p_add1, ln_node.GetExecutionProviderType(), allow_sum_output) &&
            CheckSecondAdd(graph, *p_add2, ln_node.GetExecutionProviderType()) &&
            !graph.NodeProducesGraphOutput(*p_add1) &&
            !graph.NodeProducesGraphOutput(*p_add2)) {
          matched_format = Format::Format2;
        }
      }
    }

    if (matched_format == Format::None) {
      // Format 3
      if (graph_utils::FindPath(ln_node, true,
                                {{0, 0, "Add", {7, 13, 14}, kOnnxDomain}},
                                edges, logger)) {
        p_add1 = const_cast<Node*>(&edges[0]->GetNode());

        if (CheckFirstAdd(*p_add1, ln_node.GetExecutionProviderType(), allow_sum_output) &&
            !graph.NodeProducesGraphOutput(*p_add1)) {
          matched_format = Format::Format3;
        }
      }
    }

    if (matched_format == Format::None ||
        fused_nodes.count(p_add1->Index()) > 0 ||
        (p_add2 != nullptr && matched_format != Format::Format3 && fused_nodes.count(p_add2->Index()) > 0)) {
      continue;
    }

    const bool emit_sum_output = p_add1->GetOutputEdgesCount() > 1;
    if (emit_sum_output && ElemType(p_add1->OutputDefs()[0]) != ElemType(ln_node.OutputDefs()[0])) {
      continue;
    }
    const size_t first_node_to_remove = nodes_to_remove.size();

    // SkipLayerNormalization kernel requires gamma and beta to be 1D.
    // Skip fusion if gamma or beta have more than 1 dimension.
    const NodeArg* gamma_arg = ln_node.MutableInputDefs()[1];
    const TensorShapeProto* gamma_shape = gamma_arg->Shape();
    if (gamma_shape != nullptr && gamma_shape->dim_size() != 1) {
      continue;
    }
    if (is_layer_norm && ln_node.MutableInputDefs().size() > 2) {
      const NodeArg* beta_arg = ln_node.MutableInputDefs()[2];
      const TensorShapeProto* beta_shape = beta_arg->Shape();
      if (beta_shape != nullptr && beta_shape->dim_size() != 1) {
        continue;
      }
    }

    NodeArg beta_place_holder("", nullptr);

    // Get the inputs for the new SkipLayerNormalization node.
    InlinedVector<NodeArg*> skip_layer_norm_input_defs{p_add1->MutableInputDefs()[0],
                                                       p_add1->MutableInputDefs()[1],
                                                       ln_node.MutableInputDefs()[1]};
    if (is_layer_norm) {
      skip_layer_norm_input_defs.push_back(ln_node.MutableInputDefs().size() == 2 ? &beta_place_holder
                                                                                  : ln_node.MutableInputDefs()[2]);
    }

    if (matched_format == Format::Format1) {
      skip_layer_norm_input_defs[0] = p_add2->MutableInputDefs()[0];
      skip_layer_norm_input_defs.push_back(p_add2->MutableInputDefs()[1]);
      nodes_to_remove.push_back(*p_add2);
    } else if (matched_format == Format::Format2) {
      skip_layer_norm_input_defs[1] = p_add2->MutableInputDefs()[0];
      skip_layer_norm_input_defs.push_back(p_add2->MutableInputDefs()[1]);
      nodes_to_remove.push_back(*p_add2);
    }

    nodes_to_remove.push_back(*p_add1);
    nodes_to_remove.push_back(ln_node);
    for (size_t i = first_node_to_remove; i < nodes_to_remove.size(); ++i) {
      fused_nodes.insert(nodes_to_remove[i].get().Index());
    }

    // If input types are different than output type and output type is float, insert cast node after inputs.
    for (auto& input_def : skip_layer_norm_input_defs) {
      input_def = CastToFloat(graph,
                              input_def,
                              ln_node.MutableOutputDefs()[0]->TypeAsProto()->tensor_type().elem_type(),
                              ln_node.GetExecutionProviderType());
    }

    // SkipLayerNormalization outputs: output, mean, inv_std_var, input_skip_bias_sum. LayerNormalization's
    // outputs line up with the first three; for the simplified norms only output 0 is carried over.
    InlinedVector<NodeArg*> skip_layer_norm_output_defs;
    if (is_layer_norm) {
      skip_layer_norm_output_defs.assign(ln_node.MutableOutputDefs().begin(), ln_node.MutableOutputDefs().end());
    } else {
      skip_layer_norm_output_defs.push_back(ln_node.MutableOutputDefs()[0]);
    }
    if (emit_sum_output) {
      NodeArg& empty_output = graph.GetOrCreateNodeArg("", nullptr);
      while (skip_layer_norm_output_defs.size() < 3) {
        skip_layer_norm_output_defs.push_back(&empty_output);
      }
      skip_layer_norm_output_defs.push_back(p_add1->MutableOutputDefs()[0]);
    }

    const char* fused_op_type = is_layer_norm ? "SkipLayerNormalization" : "SkipSimplifiedLayerNormalization";
    Node& skip_layer_norm_node = graph.AddNode(graph.GenerateNodeName(fused_op_type),
                                               fused_op_type,
                                               "fused SkipLayerNorm subgraphs ",
                                               skip_layer_norm_input_defs,
                                               skip_layer_norm_output_defs, {}, kMSDomain);
    // Carry over "epsilon"; when absent, use the source op's schema default (1e-5 for LayerNormalization,
    // SimplifiedLayerNormalization and RMSNormalization), which differs from the fused op's default.
    NodeAttributes ln_attrs = ln_node.GetAttributes();
    NodeAttributes::const_iterator epsilon = ln_attrs.find("epsilon");
    if (epsilon != ln_attrs.end()) {
      skip_layer_norm_node.AddAttributeProto(epsilon->second);
    } else {
      skip_layer_norm_node.AddAttribute("epsilon", kNormDefaultEpsilon);
    }
    // Assign provider to this new node. Provider should be same as the provider for old node.
    skip_layer_norm_node.SetExecutionProviderType(ln_node.GetExecutionProviderType());

    InlinedVector<NodeIndex> source_node_indices;
    source_node_indices.reserve(nodes_to_remove.size() - first_node_to_remove);
    for (size_t i = first_node_to_remove; i < nodes_to_remove.size(); ++i) {
      source_node_indices.push_back(nodes_to_remove[i].get().Index());
    }
    graph.NotifyNodeReplacement(source_node_indices, skip_layer_norm_node.Index());
  }

  for (const auto& node : nodes_to_remove) {
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.get().Index());
  }

  if (!nodes_to_remove.empty()) {
    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
