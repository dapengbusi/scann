#ifndef SCANN__SCANN_OPS_CC_SCANN_NPY_H_
#define SCANN__SCANN_OPS_CC_SCANN_NPY_H_

#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>


namespace tensorflow {
namespace scann_ops {

class ScannInterface;

class ScannExt {
 public:
  ScannExt();
  void Search(long n, const std::vector<float> &vecs, long k, std::vector<float> &distances,
              std::vector<int64_t> &labels);
  void BuildIndex(const std::vector<float>& dataset, int dimensionality, const char* config, int conf_length);
  int BuildIndex(const char* conf_str, int conf_length, const char* codebook_str, int code_length,
                 const char* partition_str, int partition_length,
		 const std::vector<float>& data_set,
                 const std::vector<int32_t>& datapoint,
                 const std::vector<uint8_t>& hashed_dataset,
                 int dimensionality);
  int WriteIndex(const char* file);
 private:
  int nprobe_ = -1;
  int training_thread_num_ = 2;
  std::shared_ptr<ScannInterface> scann_;
};

}  // namespace scann_ops
}  // namespace tensorflow

#endif

