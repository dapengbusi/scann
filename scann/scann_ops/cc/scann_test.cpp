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
#include <unistd.h>
#include <math.h>
#include <chrono>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

void norm(std::vector<float>& v) {
    double x = 0;
    for (auto i : v) {
	  x += i*i;
    }
    std::vector<float> tmp;
    for (auto i : v) {
	    tmp.push_back(i/sqrt(x));
    }
    v.swap(tmp);
}

// trim from start
static inline std::string &ltrim(std::string &s) {
  s.erase(s.begin(),
          std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
  s.erase(
      std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(),
      s.end());
  return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) { return ltrim(rtrim(s)); }

static void split(const std::string &str, char sep, std::vector<std::string> &vals) {
  std::string token;
  std::istringstream ss(str);
  while (std::getline(ss, token, sep)) {
    vals.push_back(token);
  }
}


using namespace elasticfaiss;
int main (int argc , char** argv) {
	ScannIndex index;
	std::string config = "num_children:4000,noise_shaping_threshold:0.3";
	std::string data_file_name = "./test.data";
	std::string query_file_name = "./query.data";
	if (argc == 2) {
		data_file_name = argv[0];
		query_file_name = argv[1];
	}
	std::cout << "in query: " << data_file_name << " " << query_file_name << std::endl;


	int dimension = 8 * 16;
	index.dimensionality_ = dimension;
	auto handler_v2 = [&](const std::string &file, std::vector<float>& vecs) {
		std::ifstream infile(file);
		std::istream &is = infile;

		std::string line;
		while (std::getline(is, line)) {
			std::string new_line = trim(line);
			std::vector<std::string> vals;
			split(new_line, '|', vals);
			if (vals.size() < 3) {
				continue;
			}
			std::string token;
			std::istringstream ss(vals[2]);
			std::vector<float> vector_f;
			while (std::getline(ss, token, ',')) {
				vector_f.push_back(atof(token.c_str()));
			}
			if (vector_f.size() != (size_t)dimension) {
				continue;
			}
			vecs.insert(vecs.end(), vector_f.begin(), vector_f.end());
		}
	};
	
	std::vector<float>  datas;
	std::vector<float>  query_vec;
	handler_v2(data_file_name, datas);
	handler_v2(query_file_name, query_vec);
	std::cout << "data size: " << datas.size()/dimension << " query: " << query_vec.size()/dimension << std::endl;

	//index.add_with_ids(ids, datas);
	std::vector<int64_t> ids;
	for (int i = 0; i < datas.size()/dimension; i++) {
		ids.push_back(i);
	}
	index.train(ids, datas, config, std::string("data.txt"));

	std::vector<std::set<int64_t>> result1;
	std::vector<std::set<int64_t>> result2;
	std::vector<float> distance(1000, 0);
	std::vector<int64_t> labels(1000, 0);
	auto start = std::chrono::steady_clock::now();
	for (int i=0; i < query_vec.size()/ dimension; i++) {
		std::vector<float> query {query_vec.begin() + i*dimension, query_vec.begin() + (i+1) * dimension};
		index.search(1, query, 1000, distance, labels);
		std::set<int64_t> tmp {labels.begin(), labels.end()};
		result1.push_back(tmp);
	}
	auto end = std::chrono::steady_clock::now();
	std::cout << "first : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
		

	ScannIndex index2;
	index2.load("data.txt");
	{
		std::vector<float> distance2(1000, 0);
		std::vector<int64_t> labels2(1000, 0);
		index2.add_with_ids2(ids, datas);
		auto start = std::chrono::steady_clock::now();
		for (int i=0; i < query_vec.size()/ dimension; i++) {
			std::vector<float> query {query_vec.begin() + i*dimension, query_vec.begin() + (i+1) * dimension};
			index2.search(1, query, 1000, distance2, labels2);
			std::set<int64_t> tmp {labels2.begin(), labels2.end()};
			result2.push_back(tmp);
		}
		auto end = std::chrono::steady_clock::now();
		std::cout << "second : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms "<< std::endl;
	}
	auto diff = [] (const std::set<int64_t>& x, const std::set<int64_t>& y) {
	 	int count = 0;
		for (auto i : x) {
			if (y.count(i)) count++;
		}
		return 1.0*count/x.size();
	};
	std::cout << result1.size()  << "  " << result1.front().size()  << "  " << result2.size() << " " << result2.front().size() << std::endl;
	for (int i = 0; i < result1.size(); i ++) {
		//diff(result1[i], result2[i]);
		std::cout << diff(result1[i], result2[i]) << std::endl;
	}
	return 0;
}
