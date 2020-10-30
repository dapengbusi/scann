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
#include "scann_index_test.h"

#include <cstdlib>
#include <fstream>
#include <fstream>
#include <string>
#include <vector>

#include "google/protobuf/text_format.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_no_sparse.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/scann_ops/cc/scann.h"

namespace elasticfaiss {

// ¿¿¿scann¿¿¿¿¿¿
static const std::string kProtoPrefix = "PROTO";
static const std::string kDataSpanPrefix = "NUMDATA";
static const std::string kIdMapPrefix = "IDMAP";
static const std::string kConfigPrefix = "SCANNCONFIG";
static const std::string kConfigPbName = "scann_config";
static const std::string kCodeBookPbName = "ah_codebook";
static const std::string kSerializedPartitionerPbName = "serialized_partitioner";
static const std::string kDataSetDataName = "dataset";
static const std::string kDataPointDataName = "datapoint";
static const std::string kHashedDataDataName = "hasheddata";

int ScannIndex::loadconfig(const std::string& filename) {
  std::ifstream fin(filename, std::ifstream::binary);
  if (fin.bad() || fin.fail()) {
    LOG(ERROR) << "load config error: " << filename;
    return -1;
  }

  std::string line;
  std::getline(fin, line);
  if (line.size() > kConfigPrefix.size() && line.compare(0, kConfigPrefix.size(), kConfigPrefix) == 0) {
    size_t n = line.find_first_of(std::string(":"));
    config_ = line.substr(n+1);
    return 0;
  }
  return -1;
}

int ScannIndex::saveconfig(const std::string& filename) {
  std::ofstream fout(filename, std::ofstream::binary);
  if (fout.bad() || fout.fail()) {
    LOG(ERROR) << "load config error: " << filename;
    return -1;
  }
  fout << kConfigPrefix + ":" + config_ + "\n";
  return 0;
}

int ScannIndex::init() {
  std::string file_name = "";
  return loadconfig(file_name);
}
size_t ScannIndex::size() {
  return data_set_.size();
}
bool ScannIndex::support_update() {
  return false;
}
bool ScannIndex::support_delete() {
  return false;
}

void ScannIndex::add_with_ids(const std::vector<int64_t> &ids, const std::vector<float> &vecs) {
  if (vecs.size() / ids.size() != dimensionality_) {
    LOG(INFO) << "ids size " << ids.size() << " unmatch vecs size " << vecs.size()
        << " with dimensionality " << dimensionality_;
    return;
  }
  //scann_->AddDocsWithIds(ids, vecs);
  id_map_.insert(id_map_.end(), ids.begin(), ids.end());
  data_set_.insert(data_set_.end(), vecs.begin(), vecs.end());
  return;
}

void ScannIndex::add_with_ids2(const std::vector<int64_t> &ids, const std::vector<float> &vecs) {
  if (vecs.size() / ids.size() != dimensionality_) {
    LOG(INFO) << "ids size " << ids.size() << " unmatch vecs size " << vecs.size()
        << " with dimensionality " << dimensionality_;
    return;
  }
  id_map_.insert(id_map_.end(), ids.begin(), ids.end());
  scann_->AddDocsWithIds(ids, vecs);
  return;
}

void ScannIndex::range_search(long n, const std::vector<float> &vecs, float radius,
                              std::vector<std::vector<float>> &distances,
                              std::vector<std::vector<int64_t>> &labels) {
  LOG(ERROR) << "scann index don't support range search";
  return;
}

void ScannIndex::update(const std::vector<int64_t> &ids, const std::vector<float> &vecs) {
  LOG(ERROR) << "scann don`t support update()";
}

void ScannIndex::clear() {
  scann_.reset();
  id_map_.clear();
  data_set_.clear();
}

void ScannIndex::train(const std::vector<int64_t> &ids, const std::vector<float> &vecs, const std::string& config, const std::string& output_file) {
  // train when rebuild (save)
  id_map_.insert(id_map_.end(), ids.begin(), ids.end());
  config_ = config;
  scann_->BuildIndex(vecs, dimensionality_, config.c_str(), config.length());
  scann_->WriteIndex(output_file.c_str(), false);
  return;
}

int ScannIndex::save(const std::string &filename) {
  try {
    LOG(INFO) << "add index ";
    scann_->WriteIndex(filename.c_str(), false);
    std::ofstream fout(filename, std::ofstream::binary|std::ofstream::app);
    // ¿¿¿¿¿¿
    // LOG(INFO) << "add dataset ";
    // std::string header = kDataSpanPrefix + ":" + kDataSetDataName + ":" + std::to_string(data_set_.size() * sizeof(float)) + "\n";
    // fout << header;
    // fout.write(reinterpret_cast<char*>(data_set_.data()), sizeof(float)*data_set_.size());
    // ¿¿idmap
    // LOG(INFO) << "add idmap ";
    // std::string header = kIdMapPrefix + ":" + std::to_string(sizeof(int64_t)*id_map_.size()) + "\n";
    // fout << header;
    // fout.write(reinterpret_cast<char*>(id_map_.data()), sizeof(int64_t)*id_map_.size());
    // fout << "\n";
  } catch (std::exception &e) {
    LOG(ERROR) << "Scann exception: " << e.what();
    return -1;
  }
  return 0;
}

void ScannIndex::search(long n, const std::vector<float> &vecs, long k,
                        std::vector<float> &distances, std::vector<int64_t> &labels) {

  //if (id_map_.empty()) return;

  std::vector<int64_t> index_ids;
  std::vector<float> index_distances;
  scann_->Search(n, vecs, k, index_distances, index_ids);
  if (index_ids.size() != k || index_distances.size() != k) {
    LOG(ERROR) << "scann index may has error!! search num: " << k
        << " result num : " << index_ids.size()
        << " result dist num: " << index_distances.size();
    return;
  }
  for (int i = 0; i < k; i++) {
    auto idx = index_ids[i];
    if (idx >= 0 && idx < id_map_.size()) {
      labels[i] = id_map_[idx];
    } else {
      labels[i] = -1;
    }
    distances[i] = index_distances[i];
    // LOG(INFO) << "i: " << i << ", index: " << idx << ", distance: " << index_distances[i]
    //     << ", label: " << labels[i];
  }
  return;
}

int ScannIndex::load(const std::string& file_name) {
	LOG(INFO) << "load: ";
  std::string conf_str;
  std::string codebook_str;
  std::string partition_str;

  std::ifstream fin(file_name, std::ifstream::binary);
  std::vector<float> dataset_tmp;

  for (std::string line; std::getline(fin, line);) {
    if (line.size() > kConfigPrefix.size() && line.compare(0, kConfigPrefix.size(), kConfigPrefix) == 0) {
      size_t n = line.find_first_of(std::string(":"));
      config_ = line.substr(n+1);
    } else if (line.size() > kProtoPrefix.size() && line.compare(0, kProtoPrefix.size(), kProtoPrefix) == 0) {
      /*
      vector<string> str_list;
      butil::SplitString(line, ':', &str_list);
      std::string pb_name = str_list[1];
      int length = std::atoi(str_list[2].c_str());
      */
      size_t n = line.find_first_of(std::string(":"));
      std::string sub_str = line.substr(n+1);
      n = sub_str.find_first_of(std::string(":"));
      std::string pb_name = sub_str.substr(0, n);
      int64_t length = std::atoi(sub_str.substr(n+1).c_str());

      if (length > 0) {
        std::string str(length, '\0');
        fin.read(&str[0], length);
        if (fin.bad() || fin.fail()) {
          LOG(ERROR) << "Failed to load pb buffer";
          return -1;
        }
        if (pb_name == kConfigPbName) {
          conf_str.swap(str);
        } else if (pb_name == kCodeBookPbName) {
          codebook_str.swap(str);
        } else if (pb_name == kSerializedPartitionerPbName) {
          partition_str.swap(str);
        }
      }
    } else if (line.size() > kDataSpanPrefix.size() && line.compare(0, kDataSpanPrefix.size(), kDataSpanPrefix) == 0) {
      /*
      vector<string> str_list;
      butil::SplitString(line, ':', &str_list);
      std::string data_name = str_list[1];
      int length = std::atoi(str_list[2].c_str());
      */
      size_t n = line.find_first_of(std::string(":"));
      std::string sub_str = line.substr(n+1);
      n = sub_str.find_first_of(std::string(":"));
      std::string data_name = sub_str.substr(0, n);
      int64_t length = std::atoi(sub_str.substr(n+1).c_str());

      if (length > 0) {
        std::vector<char> buffer(length);
        fin.read(buffer.data(), length);
        if (fin.bad() || fin.fail()) {
          LOG(ERROR) << "Failed to parse " << data_name;
          return -1;
        }
        if (data_name == kDataSetDataName) {
          //data_set_.insert(data_set_.end(), (float*)(buffer.data()), (float*)(buffer.data() + length));
          //dataset_tmp.insert(dataset_tmp.end(), (float*)(buffer.data()), (float*)(buffer.data() + length));
        } else if (data_name == kDataPointDataName) {
          datapoint_to_token_.insert(datapoint_to_token_.end(), (int32_t*)(buffer.data()), (int32_t*)(buffer.data() + length));
        } else if (data_name == kHashedDataDataName) {
          hashed_dataset_.insert(hashed_dataset_.end(), (uint8_t*)(buffer.data()), (uint8_t*)(buffer.data() + length));
        }
      }
    } else if (line.size() > kIdMapPrefix.size() && line.compare(0, kIdMapPrefix.size(), kIdMapPrefix) == 0) {
      size_t n = line.find_first_of(std::string(":"));
      std::string sub_str = line.substr(n+1);
      int64_t length = std::atoi(sub_str.c_str());
      if (length > 0) {
        std::vector<char> buffer(length);
        fin.read(buffer.data(), length);
        if (fin.bad() || fin.fail()) {
          LOG(ERROR) << "Failed to parse " << sub_str;
          return -1;
        }
        id_map_.insert(id_map_.end(), (int64_t*)(buffer.data()), (int64_t*)(buffer.data() + length));
      }
    }
  }

  scann_->BuildIndex(conf_str.c_str(), conf_str.length(),
		  codebook_str.c_str(), codebook_str.length(),
		  partition_str.c_str(), partition_str.length(),
		  //data_set_, datapoint_to_token_, hashed_dataset_,
		  dataset_tmp, datapoint_to_token_, hashed_dataset_,
		  dimensionality_);
  return 0;
}

ScannIndex::ScannIndex() {
    scann_ = std::make_unique<::tensorflow::scann_ops::ScannExt>();
}
ScannIndex::~ScannIndex() { clear(); }

}  // namespace elasticfaiss
