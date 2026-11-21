// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/skip_layer_norm_fusion.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/graph_utils.h"
#include "float.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

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

static bool HasRank(const NodeArg& arg, int rank) {
  const TensorShapeProto* shape = arg.Shape();
  return shape != nullptr && shape->dim_size() == rank;
}

static bool HasRankAtMostOne(const NodeArg& arg) {
  const TensorShapeProto* shape = arg.Shape();
  return shape != nullptr && shape->dim_size() <= 1;
}

// Returns true if `arg` is computed from `source` through a chain of single-data-input ops that keep every dimension
// except the last one unchanged for inputs of rank >= 1:
//   - MatMul with a 2D second input,
//   - Add/Sub/Mul/Div and BiasGelu whose other input has rank <= 1,
//   - unary elementwise activations.
// None of these ops can raise the rank above max(rank, 1), so when `source` and `arg` both have rank 3 the
// leading dimensions of `arg` provably equal those of `source`, even if they are unnamed in shape inference
// (e.g. the FFN residual in transformer exports with dynamic batch/sequence dims).
static bool HasSameLeadingDimsAs(const Graph& graph, const NodeArg* arg, const NodeArg* source) {
  constexpr int kMaxChainLength = 16;
  for (int i = 0; i < kMaxChainLength; ++i) {
    if (arg == source) {
      return true;
    }

    const Node* node = graph.GetProducerNode(arg->Name());
    if (node == nullptr) {
      return false;
    }

    const auto& inputs = node->InputDefs();
    const NodeArg* next = nullptr;
    if (graph_utils::IsSupportedOptypeVersionAndDomain(*node, "MatMul", {1, 9, 13})) {
      if (HasRank(*inputs[1], 2)) {
        next = inputs[0];
      }
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Add", {7, 13, 14}) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Sub", {7, 13, 14}) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Mul", {7, 13, 14}) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Div", {7, 13, 14})) {
      if (HasRankAtMostOne(*inputs[1])) {
        next = inputs[0];
      } else if (HasRankAtMostOne(*inputs[0])) {
        next = inputs[1];
      }
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*node, "BiasGelu", {1}, kMSDomain)) {
      if (HasRankAtMostOne(*inputs[1])) {
        next = inputs[0];
      }
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Gelu", {1}, kMSDomain) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Gelu", {20}) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Relu", {6, 13, 14}) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Tanh", {6, 13}) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Sigmoid", {6, 13}) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Erf", {9, 13})) {
      next = inputs[0];
    }

    if (next == nullptr) {
      return false;
    }
    arg = next;
  }

  return false;
}

static bool CheckFirstAdd(const Graph& graph, Node& add, ProviderType providertype) {
  if (providertype != add.GetExecutionProviderType() ||
      !IsSupportedDataType(add) ||
      add.GetOutputEdgesCount() != 1) {
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
  if (is_valid_input) {
    return true;
  }

  // Dynamic-shape exports usually leave batch/sequence dims unnamed after MatMul/Reshape, so they cannot be compared
  // above. The SkipLayerNormalization kernel does not support broadcasting the batch dim of its first input, so only
  // fuse when the hidden dims are known and equal and one input is provably derived from the other with the same
  // leading dims.
  const auto& hidden1 = add_input1_shape->dim(2);
  const auto& hidden2 = add_input2_shape->dim(2);
  if (!utils::HasDimValue(hidden1) || !utils::HasDimValue(hidden2) ||
      hidden1.dim_value() != hidden2.dim_value()) {
    return false;
  }

  const NodeArg* input1 = add.InputDefs()[0];
  const NodeArg* input2 = add.InputDefs()[1];
  return HasSameLeadingDimsAs(graph, input1, input2) || HasSameLeadingDimsAs(graph, input2, input1);
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

/**
Skip Layer Normalization will fuse Add + LayerNormalization into one node, and another Add if applicable

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
  for (auto node_index : node_topology_list) {
    Node* p_layernorm = graph.GetNode(node_index);
    if (p_layernorm == nullptr)
      continue;  // node was removed in an earlier fusion.

    Node& ln_node = *p_layernorm;
    ORT_RETURN_IF_ERROR(Recurse(ln_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(ln_node, "LayerNormalization", {1, 17}) ||
        !graph_utils::IsSupportedProvider(ln_node, GetCompatibleExecutionProviders()) ||
        !IsSupportedDataType(ln_node)) {
      continue;
    }

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

      if (CheckFirstAdd(graph, *p_add1, ln_node.GetExecutionProviderType()) &&
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

        if (CheckFirstAdd(graph, *p_add1, ln_node.GetExecutionProviderType()) &&
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

        if (CheckFirstAdd(graph, *p_add1, ln_node.GetExecutionProviderType()) &&
            !graph.NodeProducesGraphOutput(*p_add1)) {
          matched_format = Format::Format3;
        }
      }
    }

    if (matched_format == Format::None) {
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
    if (ln_node.MutableInputDefs().size() > 2) {
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
                                                       ln_node.MutableInputDefs()[1],
                                                       ln_node.MutableInputDefs().size() == 2 ? &beta_place_holder : ln_node.MutableInputDefs()[2]};

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

    // If input types are different than output type and output type is float, insert cast node after inputs.
    for (auto& input_def : skip_layer_norm_input_defs) {
      input_def = CastToFloat(graph,
                              input_def,
                              ln_node.MutableOutputDefs()[0]->TypeAsProto()->tensor_type().elem_type(),
                              ln_node.GetExecutionProviderType());
    }

    Node& skip_layer_norm_node = graph.AddNode(graph.GenerateNodeName("SkipLayerNormalization"),
                                               "SkipLayerNormalization",
                                               "fused SkipLayerNorm subgraphs ",
                                               skip_layer_norm_input_defs,
                                               ln_node.MutableOutputDefs(), {}, kMSDomain);
    // Get attribute "epsilon" from "LayerNormalization" node if available. Else, default value
    // will be used.
    NodeAttributes ln_attrs = ln_node.GetAttributes();
    NodeAttributes::const_iterator epsilon = ln_attrs.find("epsilon");
    if (epsilon != ln_attrs.end()) {
      skip_layer_norm_node.AddAttributeProto(epsilon->second);
    } else {
      skip_layer_norm_node.AddAttribute("epsilon", contrib::kDefaultSkipLayerNormEpsilon);
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
