/*
 *Copyright (c) 2018, Tencent. All rights reserved.
 *
 *Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of elasticfaiss nor the names of its contributors may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 *THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
 *BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 *THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "scann/scann_ops/cc/scann_build.h"

#include <cstdlib>
#include <fstream>
#include <map>

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_no_sparse.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/threads.h"
#include "scann/proto/scann.pb.h"
#include "scann/proto/input_output.pb.h"


namespace tensorflow {
namespace scann_ops {

static void split_string(const std::string& s, std::vector<std::string>& v, const std::string& c) {
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while (std::string::npos != pos2) {
    v.push_back(s.substr(pos1, pos2 - pos1));
    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if (pos1 != s.length()) v.push_back(s.substr(pos1));
}

static void RuntimeErrorIfNotOk(const char* prefix, const Status& status) {
  if (!status.ok()) {
    std::string msg = prefix + std::string(status.error_message());
    throw std::runtime_error(msg);
  }
}

ScannExt::ScannExt() {
  scann_ = std::make_shared<ScannInterface>();
}

void ScannExt::BuildIndex(const std::vector<float>& dataset, int dimensionality, const char* config, int conf_length) {
  std::string config_str = std::string(config, conf_length);

  std::map<std::string, std::string> conf_map;
  std::vector<std::string> conf_pairs;
  split_string(config_str, conf_pairs, std::string(","));
  for (auto conf : conf_pairs) {
    std::vector<std::string> conf_pair;
    split_string(conf, conf_pair, std::string(":"));
    if (conf_pair.size() == 2) {
      conf_map[conf_pair.front()] = conf_pair.back();
    }
  }

  ScannConfig scann_conf;
  {
    scann_conf.set_num_neighbors(10);
    scann_conf.mutable_distance_measure()->set_distance_measure("DotProductDistance");
    scann_conf.mutable_input_output()->set_in_memory_data_type(InputOutputConfig::FLOAT);
    // partitioning
    scann_conf.mutable_partitioning()->set_num_children(4000);
    scann_conf.mutable_partitioning()->set_max_clustering_iterations(10);
    scann_conf.mutable_partitioning()->set_min_cluster_size(50);
    scann_conf.mutable_partitioning()->mutable_partitioning_distance()->set_distance_measure("SquaredL2Distance");
    scann_conf.mutable_partitioning()->mutable_query_spilling()->set_spilling_type(QuerySpillingConfig::FIXED_NUMBER_OF_CENTERS);
    scann_conf.mutable_partitioning()->mutable_query_spilling()->set_max_spill_centers(1000);
    scann_conf.mutable_partitioning()->set_partitioning_type(PartitioningConfig::GENERIC);
    scann_conf.mutable_partitioning()->mutable_query_tokenization_distance_override()->set_distance_measure("DotProductDistance");
    scann_conf.mutable_partitioning()->set_query_tokenization_type(PartitioningConfig::FLOAT);
    scann_conf.mutable_partitioning()->set_expected_sample_size(60000);
    // hash
    scann_conf.mutable_hash()->mutable_asymmetric_hash()->set_noise_shaping_threshold(0.2);
    scann_conf.mutable_hash()->mutable_asymmetric_hash()->mutable_projection()->set_input_dim(dimensionality);
    int dims_per_block = 2;
    if (dimensionality % dims_per_block == 0) {
      scann_conf.mutable_hash()->mutable_asymmetric_hash()->mutable_projection()->set_projection_type(ProjectionConfig::CHUNK);
      scann_conf.mutable_hash()->mutable_asymmetric_hash()->mutable_projection()->set_num_blocks(dimensionality/dims_per_block);
      scann_conf.mutable_hash()->mutable_asymmetric_hash()->mutable_projection()->set_num_dims_per_block(dims_per_block);
    } else {
      scann_conf.mutable_hash()->mutable_asymmetric_hash()->mutable_projection()->set_projection_type(ProjectionConfig::VARIABLE_CHUNK);
      auto vb = scann_conf.mutable_hash()->mutable_asymmetric_hash()->mutable_projection()->add_variable_blocks();
      vb->set_num_blocks(dimensionality / dims_per_block);
      vb->set_num_dims_per_block(dims_per_block);
      vb = scann_conf.mutable_hash()->mutable_asymmetric_hash()->mutable_projection()->add_variable_blocks();
      vb->set_num_blocks(1);
      vb->set_num_dims_per_block(dimensionality % dims_per_block);
    }

    scann_conf.mutable_hash()->mutable_asymmetric_hash()->set_num_clusters_per_block(16);
    scann_conf.mutable_hash()->mutable_asymmetric_hash()->set_max_clustering_iterations(10);
    scann_conf.mutable_hash()->mutable_asymmetric_hash()->mutable_quantization_distance()->set_distance_measure("SquaredL2Distance");
    scann_conf.mutable_hash()->mutable_asymmetric_hash()->set_min_cluster_size(100);
    scann_conf.mutable_hash()->mutable_asymmetric_hash()->set_lookup_type(AsymmetricHasherConfig::INT8_LUT16);
    scann_conf.mutable_hash()->mutable_asymmetric_hash()->set_use_residual_quantization(true);
    scann_conf.mutable_hash()->mutable_asymmetric_hash()->set_expected_sample_size(100000);
    // exact_reordering
    scann_conf.mutable_exact_reordering()->set_approx_num_neighbors(1000);
    scann_conf.mutable_exact_reordering()->mutable_fixed_point()->set_enabled(false);
  }

  //聚类数
  if (conf_map.count("num_children")) {
    int num_children = std::atoi(conf_map["num_children"].c_str());
    num_children = num_children == 0 ? 4000 : num_children;
    scann_conf.mutable_partitioning()->set_num_children(num_children);
  }
  if (conf_map.count("noise_shaping_threshold")) {
    float noise_shaping = std::atof(conf_map["noise_shaping_threshold"].c_str());
    noise_shaping = noise_shaping == 0.0 ? 1 : noise_shaping;
    scann_conf.mutable_hash()->mutable_asymmetric_hash()->set_noise_shaping_threshold(noise_shaping);
  }
  // 距离计算方式
  if (conf_map.count("distance_measure")) {
    if (conf_map["distance_measure"] == "dot_product") {
      scann_conf.mutable_distance_measure()->set_distance_measure("DotProductDistance");
      scann_conf.mutable_partitioning()->mutable_query_tokenization_distance_override()->set_distance_measure("DotProductDistance");
    } else if (conf_map["distance_measure"] == "squared_l2") {
      scann_conf.mutable_distance_measure()->set_distance_measure("SquaredL2Distance");
      scann_conf.mutable_hash()->mutable_asymmetric_hash()->set_use_residual_quantization(false);
      scann_conf.mutable_partitioning()->mutable_query_tokenization_distance_override()->set_distance_measure("SquaredL2Distance");
    }
  }
  // train_sample_ratio
  if (conf_map.count("train_sample_ratio")) {
    float train_sample_ratio = std::atof(conf_map["train_sample_ratio"].c_str());
    int train_sample_size = dataset.size()/dimensionality * train_sample_ratio;
    scann_conf.mutable_partitioning()->set_expected_sample_size(train_sample_size);
  }
  // 预估最大搜索数
  if (conf_map.count("max_search_num")) {
    int max_num = std::atoi(conf_map["max_search_num"].c_str());
    scann_conf.mutable_exact_reordering()->set_approx_num_neighbors(2*max_num);
  }
  if (conf_map.count("nprobe")) {
    nprobe_ = std::atoi(conf_map["nprobe"].c_str());
    scann_conf.mutable_partitioning()->mutable_query_spilling()->set_max_spill_centers(nprobe_);
    scann_conf.set_num_neighbors(500);
  }
  if (conf_map.count("train_thread_num")) {
    training_thread_num_ = atoi(conf_map["train_thread_num"].c_str());
  }

  LOG(ERROR) << "scann config: " << scann_conf.DebugString();
  ConstSpan<float> ds_span = absl::MakeConstSpan(dataset);
  auto status = scann_->Initialize(ds_span, dimensionality, scann_conf, training_thread_num_);
  return;
}

int ScannExt::WriteIndex(const char* filename) {
  return scann_->WriteIndex(std::string(filename));
}


int ScannExt::BuildIndex(const char* conf, int conf_length, const char* codebook, int code_length,
		const char* partition, int partition_length,
		const std::vector<float>& data_set,
		const std::vector<int32_t>& datapoint,
		const std::vector<uint8_t>& hashed_dataset,
		int dimensionality) {
  std::string conf_str = std::string(conf, conf_length);
  std::string codebook_str = std::string(codebook, code_length);
  std::string partition_str = std::string(partition, partition_length);

  ScannConfig config;
  config.ParseFromString(conf_str);
  SingleMachineFactoryOptions opts;
  opts.ah_codebook = std::make_shared<CentersForAllSubspaces>();
  opts.ah_codebook->ParseFromString(codebook_str);
  opts.serialized_partitioner = std::make_shared<SerializedPartitioner>();
  opts.serialized_partitioner->ParseFromString(partition_str);

  ConstSpan<float> ds_span = absl::MakeConstSpan(data_set);
  ConstSpan<int32_t> dp_span = absl::MakeConstSpan(datapoint);
  ConstSpan<uint8_t> hd_span = absl::MakeConstSpan(hashed_dataset);

  scann_->Initialize(config, opts, ds_span, dp_span, hd_span, dimensionality);
  return 0;
}

void ScannExt::Search(long n, const std::vector<float> &vecs, long k,
                        std::vector<float> &distances, std::vector<int64_t> &labels) {
  if (n == 1) {
    DatapointPtr<float> ptr(nullptr, vecs.data(), vecs.size(), vecs.size());
    NNResultsVector res;
    auto status = scann_->Search(ptr, &res, k, -1, nprobe_);
    RuntimeErrorIfNotOk("Error during search: ", status);

    distances.resize(k);
    labels.resize(k);
    scann_->ReshapeNNResult(res, labels.data(), distances.data());
  } else {
    // batch接口
  }
  return;
}

}  // namespace scann_ops
}  // namespace tensorflow
