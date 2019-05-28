#include <iostream>
#include "sfm.h"
#include <vector>
#include <set>
#include <io.h>
#include <fstream>
#include <string>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
using namespace std;
using namespace cv;

void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions,
	vector<Point3d>& structure, vector<Vec3b>& colors) {
	int n = (int)rotations.size();

	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << (int)structure.size();

	fs << "Rotations" << "[";
	for (size_t i = 0; i < n; ++i) {
		fs << rotations[i];
	}
	fs << "]";

	fs << "Motions" << "[";
	for (size_t i = 0; i < n; i++) {
		fs << motions[i];
	}
	fs << "]";

	fs << "Points" << "[";
	for (size_t i = 0; i < structure.size(); ++i) {
		fs << structure[i];
	}
	fs << "]";

	fs << "Colors" << "[";
	for (size_t i = 0; i < colors.size(); ++i) {
		fs << colors[i];
	}
	fs << "]";

	fs.release();

}

void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector <vector<Vec3b>>& colors_for_all
) {
	key_points_for_all.clear();
	descriptor_for_all.clear();
	Mat image;

	// 读取图像，获取图像特征点并保存
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);
	//Ptr<ORB> orb = ORB::create();
	//Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create();

	for (auto it = image_names.begin(); it != image_names.end(); ++it) {
		image = imread(*it);
		if (image.empty()) {
			continue;
		}
		cout << "Extracting features: " << *it;

		vector<KeyPoint> key_points;
		Mat descriptor;
		// 偶尔出现内存分配失败的错误  Detects keypoints and computes the descriptors
		// sift->detectAndCompute(image, noArray(), key_points, descriptor);
		//orb->detect(image, key_points);
		//orb->compute(image, key_points, descriptor);
		//surf->detect(image, key_points);
		//surf->compute(image, key_points, descriptor);
		sift->detect(image, key_points);
		sift->compute(image, key_points, descriptor);

		//F:\DATASET\qinghuamen\qhm-small-tnt\keypoints
		string kp_path = "F:\\DATASET\\qinghuamen\\qhm-small-tnt\\fountain_keypoints\\";
		Mat out;
		drawKeypoints(image, key_points, out);
		imwrite(kp_path + get_filename_from_path(*it), out);

		// 特征点过少，则排除该图像
		if (key_points.size() <= 10) {
			continue;
		}

		cout << " :" << key_points.size() << endl;
		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);

		vector<Vec3b> colors(key_points.size());	// 三通道 存放该位置三通道颜色
		for (int i = 0; i < key_points.size(); ++i) {
			Point2f& p = key_points[i].pt;
			if (p.x <= image.rows && p.y <= image.cols)
				colors[i] = image.at<Vec3b>(p.x, p.y);
		}

		colors_for_all.push_back(colors);
	}
}

void match_features(vector<Mat>& descriptor_for_all, 
	vector<vector<DMatch>>& matches_for_all, 
	vector<vector<int>>& img_add_order) {
	//vector<vector<int>> img_add_order;
	vector<int> tmp_img_pair;
	matches_for_all.clear();
	// n个图像，两两匹配有 n-1 对匹配
	//先选择一对匹配值最大的照片用于init
	int img1_index = 0, img2_index = 1, max_match_size = 10;
	vector<DMatch> max_match;
	int match_size_table[100][100];
	for (size_t i = 0; i < descriptor_for_all.size(); i++) {
		for (size_t j = i + 1; j < descriptor_for_all.size(); j++) {
			cout << "img" << i << "---" << "img" << j << " :";
			vector<DMatch> matches;
			match_features(descriptor_for_all[i], descriptor_for_all[j], matches);
			cout << matches.size() << endl;
			match_size_table[i][j] = matches.size();
			match_size_table[j][i] = match_size_table[i][j];
			if (matches.size() > max_match_size) {
				max_match_size = matches.size();
				max_match = matches;
				img1_index = i;
				img2_index = j;
			}
		}
	}
	matches_for_all.push_back(max_match);
	cout << "max match is :" << img1_index << "---" << img2_index << ":" << max_match_size << endl;
	tmp_img_pair.clear();
	tmp_img_pair.push_back(img1_index);
	tmp_img_pair.push_back(img2_index);
	img_add_order.push_back(tmp_img_pair);

	//再决定其他照片的加入顺序
	deque<int> construct_deque;
	construct_deque.push_front(img1_index);
	construct_deque.push_back(img2_index);
	vector<int> other_index_list;
	for (size_t i = 0; i < descriptor_for_all.size(); i++) {
		if (i != img1_index&&i != img2_index) {
			other_index_list.push_back(i);
		}
	}
	cout << other_index_list.size() << endl;

	while (other_index_list.size() > 0) {
		int max_front_match = 1, max_back_match = 1;
		int max_front_match_index = other_index_list[0];
		int max_back_match_index = other_index_list[0];
		for each (int other_index in other_index_list) {
			//queue front
			if (match_size_table[construct_deque.front()][other_index] > max_front_match) {
				max_front_match = match_size_table[construct_deque.front()][other_index];
				max_front_match_index = other_index;
			}
			if (match_size_table[construct_deque.back()][other_index] > max_back_match) {
				max_back_match = match_size_table[construct_deque.back()][other_index];
				max_back_match_index = other_index;
			}
		}
		if (max_front_match > max_back_match) {
			cout << "add image" << max_front_match_index << " to match " << construct_deque.front() << endl;
			tmp_img_pair.clear();
			tmp_img_pair.push_back(construct_deque.front());
			tmp_img_pair.push_back(max_front_match_index);
			img_add_order.push_back(tmp_img_pair);

			//match_add 需要和上面的pair保持一致
			vector<DMatch> match_add;
			match_features(descriptor_for_all[construct_deque.front()], descriptor_for_all[max_front_match_index], match_add);
			cout << "match size is " << match_add.size() << endl;
			matches_for_all.push_back(match_add);

			construct_deque.push_front(max_front_match_index);
			for (vector<int>::iterator it = other_index_list.begin(); it != other_index_list.end();) {
				if (*it == max_front_match_index) {
					it = other_index_list.erase(it);
				} else {
					++it;
				}
			}

		} else {
			cout << "add image" << max_back_match_index << " to match " << construct_deque.back() << endl;
			tmp_img_pair.clear();
			tmp_img_pair.push_back(construct_deque.back());
			tmp_img_pair.push_back(max_back_match_index);
			img_add_order.push_back(tmp_img_pair);

			//match_add 需要和上面的pair保持一致
			vector<DMatch> match_add;
			match_features(descriptor_for_all[construct_deque.back()], descriptor_for_all[max_back_match_index], match_add);
			cout << "match size is " << match_add.size() << endl;
			matches_for_all.push_back(match_add);

			construct_deque.push_back(max_back_match_index);
			for (vector<int>::iterator it = other_index_list.begin(); it != other_index_list.end();) {
				if (*it == max_back_match_index) {
					it = other_index_list.erase(it);
				} else {
					++it;
				}
			}
		}
	}
	/*
	for (int i = 0; i < descriptor_for_all.size() - 1; ++i)
	{
	cout << "Matching images " << i << " - " << i + 1 ;
	vector<DMatch> matches;
	match_features(descriptor_for_all[i], descriptor_for_all[i + 1], matches);
	matches_for_all.push_back(matches);
	cout << " :" << matches.size() << endl;
	}*/
}

//按匹配数排序添加顺序
void match_features(vector<Mat>& descriptor_for_all,
	vector<vector<KeyPoint>> keyPoints_for_all,
	vector<DMatch> match_table[100][100],
	vector<vector<DMatch>>& matches_for_all,
	vector<vector<int>>& img_add_order) {
	//vector<vector<int>> img_add_order;
	vector<int> tmp_img_pair;
	matches_for_all.clear();
	// n个图像，两两匹配有 n-1 对匹配
	//先选择一对匹配值最大的照片用于init
	int img1_index = 0, img2_index = 1, max_match_size = 10;
	vector<DMatch> max_match;
	int match_size_table[100][100];
	

	for (size_t i = 0; i < descriptor_for_all.size(); i++) {
		for (size_t j = i + 1; j < descriptor_for_all.size(); j++) {
			cout << "img" << i << "---" << "img" << j << " :";
			vector<DMatch> matches;
			match_features(descriptor_for_all[i], descriptor_for_all[j],keyPoints_for_all[i],keyPoints_for_all[j], matches);
			cout << matches.size() << endl;
			match_size_table[i][j] = matches.size();
			match_table[i][j] = matches;
			if (matches.size() > max_match_size) {
				max_match_size = matches.size();
				max_match = matches;
				img1_index = i;
				img2_index = j;
			}

			cout << "img" << j << "---" << "img" << i << " :";
			vector<DMatch> matches2;
			match_features( descriptor_for_all[j], descriptor_for_all[i],  keyPoints_for_all[j], keyPoints_for_all[i], matches2);
			cout << matches2.size() << endl;
			match_size_table[j][i] = matches2.size();
			match_table[j][i] = matches2;
			if (matches2.size() > max_match_size) {
				max_match_size = matches2.size();
				max_match = matches2;
				img1_index = j;
				img2_index = i;
			}
			
		}
	}
	matches_for_all.push_back(max_match);
	cout << "max match is :" << img1_index << "---" << img2_index << ":" << max_match_size << endl;
	tmp_img_pair.clear();
	tmp_img_pair.push_back(img1_index);
	tmp_img_pair.push_back(img2_index);
	img_add_order.push_back(tmp_img_pair);

//再决定其他照片的加入顺序：通过与已参与重建的照片能匹配最多的照片
	//deque<int> construct_deque;
	vector<int> construct_list;
	//construct_deque.push_front(img1_index);
	//construct_deque.push_back(img2_index);
	construct_list.push_back(img1_index);
	construct_list.push_back(img2_index);

	vector<int> other_index_list;
	for (size_t i = 0; i < descriptor_for_all.size(); i++) {
		if (i != img1_index&&i != img2_index) {
			other_index_list.push_back(i);
		}
	}
	//cout << other_index_list.size() << endl;

	while (other_index_list.size() > 0) {
		//int max_front_match = 1, max_back_match = 1;
		//int max_other_match = 1;
		//int max_front_match_index = other_index_list[0];
		//int max_back_match_index = other_index_list[0];
		int max_other_match_index = other_index_list[0];
		int max_construct_match_index = construct_list[0];
		int max_other_match = 1;

		//for each (int construct_index in construct_list) {
		for (int i = 0; i < construct_list.size();i++) {
			int construct_index = construct_list[i];
			for each (int other_index in other_index_list) {
				if (match_size_table[construct_index][other_index]>max_other_match) {
					max_other_match = match_size_table[construct_index][other_index];
					max_other_match_index = other_index;
					max_construct_match_index = construct_index;
				}
				/*if (match_size_table[other_index][construct_index] > max_other_match) {
				max_other_match = match_size_table[other_index][construct_index];
				max_other_match_index = other_index;
				max_construct_match_index = construct_index;
				}*/
			}
			cout << "max other match:" << max_other_match << endl;
		}
		
		cout << "add image" << max_other_match_index << " to match " << max_construct_match_index << endl;
		tmp_img_pair.clear();
		tmp_img_pair.push_back(max_construct_match_index);
		tmp_img_pair.push_back(max_other_match_index);
		img_add_order.push_back(tmp_img_pair);

		cout << "match size is " << match_size_table[max_construct_match_index][max_other_match_index] << endl;
		matches_for_all.push_back(match_table[max_construct_match_index][max_other_match_index]);

		construct_list.push_back(max_other_match_index);
		for (vector<int>::iterator it = other_index_list.begin(); it != other_index_list.end();) {
			if (*it == max_other_match_index) {
				it = other_index_list.erase(it);
			} else {
				++it;
			}
		}
	}
}

//按与已参与重建的关键点匹配的数添加顺序
//void match_features(vector<Mat>& descriptor_for_all,
//	vector<vector<KeyPoint>> keyPoints_for_all,
//	vector<vector<DMatch>>& matches_for_all,
//	vector<vector<int>>& img_add_order) {
//	//vector<vector<int>> img_add_order;
//	vector<int> tmp_img_pair;
//	matches_for_all.clear();
//	// n个图像，两两匹配有 n-1 对匹配
//	//先选择一对匹配值最大的照片用于init
//	int img1_index = 0, img2_index = 1, max_match_size = 10;
//	vector<DMatch> max_match;
//	int match_size_table[100][100];
//	vector<DMatch> match_table[100][100];
//
//
//	//为了new match add method
//	vector<set<int>> matched_keypoints_idx;//每张照片参与重建的关键点下标
//	matched_keypoints_idx.resize(keyPoints_for_all.size());
//
//
//	for (size_t i = 0; i < descriptor_for_all.size(); i++) {
//		for (size_t j = i + 1; j < descriptor_for_all.size(); j++) {
//			cout << "img" << i << "---" << "img" << j << " :";
//			vector<DMatch> matches;
//			match_features(descriptor_for_all[i], descriptor_for_all[j], keyPoints_for_all[i], keyPoints_for_all[j], matches);
//			cout << matches.size() << endl;
//			match_size_table[i][j] = matches.size();
//			match_table[i][j] = matches;
//
//			vector<DMatch> matches2;
//			match_features( descriptor_for_all[j], descriptor_for_all[i],  keyPoints_for_all[j], keyPoints_for_all[i], matches2);
//			match_size_table[j][i] = matches2.size();
//			match_table[j][i] = matches2;
//
//			if (matches.size() > max_match_size) {
//				max_match_size = matches.size();
//				max_match = matches;
//				img1_index = i;
//				img2_index = j;
//			}
//		}
//	}
//	matches_for_all.push_back(max_match);
//	cout << "max match is :" << img1_index << "---" << img2_index << ":" << max_match_size << endl;
//	tmp_img_pair.clear();
//	tmp_img_pair.push_back(img1_index);
//	tmp_img_pair.push_back(img2_index);
//	img_add_order.push_back(tmp_img_pair);
//
//	//for new match method
//	for (int i = 0; i < max_match.size(); i++) {
//		matched_keypoints_idx[img1_index].insert(max_match[i].queryIdx);
//	}
//
//	cout << img1_index << " construct keypoints size:" << matched_keypoints_idx[img1_index].size() << endl;
//
//	for (int i = 0; i < max_match.size(); i++) {
//		matched_keypoints_idx[img2_index].insert(max_match[i].trainIdx);
//	}
//
//	cout << img2_index << " construct keypoints size:" << matched_keypoints_idx[img2_index].size() << endl;
//
//
//	//再决定其他照片的加入顺序：通过与已参与重建的照片能匹配最多的照片
//	deque<int> construct_deque;
//	construct_deque.push_front(img1_index);
//	construct_deque.push_back(img2_index);
//	vector<int> other_index_list;
//	for (size_t i = 0; i < descriptor_for_all.size(); i++) {
//		if (i != img1_index&&i != img2_index) {
//			other_index_list.push_back(i);
//		}
//	}
//	cout << other_index_list.size() << endl;
//
//	while (other_index_list.size() > 0) {
//		int max_front_match = 1, max_back_match = 1;
//		int max_front_match_index = other_index_list[0];
//		int max_back_match_index = other_index_list[0];
//		for each (int other_index in other_index_list) {
//			//queue front
//			/*if (match_size_table[construct_deque.front()][other_index] > max_front_match) {
//				max_front_match = match_size_table[construct_deque.front()][other_index];
//				max_front_match_index = other_index;
//			}*/
//			int tmp_match_size = 0;
//			int tmp_matched_kp_idx;
//			for (int i = 0; i < match_table[construct_deque.front()][other_index].size();i++) {
//				tmp_matched_kp_idx = match_table[construct_deque.front()][other_index][i].trainIdx;
//				if (matched_keypoints_idx[construct_deque.front()].count(tmp_matched_kp_idx)) {
//					tmp_match_size++;
//				}
//			}
//			if (tmp_match_size>max_front_match) {
//				max_front_match = tmp_match_size;
//				max_front_match_index = other_index;
//			}
//
//			//queue back
//			/*if (match_size_table[construct_deque.back()][other_index] > max_back_match) {
//				max_back_match = match_size_table[construct_deque.back()][other_index];
//				max_back_match_index = other_index;
//			}*/
//			tmp_match_size = 0;
//			for (int i = 0; i < match_table[construct_deque.back()][other_index].size(); i++) {
//				tmp_matched_kp_idx = match_table[construct_deque.back()][other_index][i].trainIdx;
//				if (matched_keypoints_idx[construct_deque.back()].count(tmp_matched_kp_idx)) {
//					tmp_match_size++;
//				}
//			}
//			if (tmp_match_size > max_back_match) {
//				max_back_match = tmp_match_size;
//				max_back_match_index = other_index;
//			}
//		}
//		if (max_front_match > max_back_match) {
//			cout << "add image" << max_front_match_index << " to match " << construct_deque.front() << endl;
//			tmp_img_pair.clear();
//			tmp_img_pair.push_back(construct_deque.front());
//			tmp_img_pair.push_back(max_front_match_index);
//			img_add_order.push_back(tmp_img_pair);
//
//			//for new match method 
//			for (int i = 0; i < match_table[construct_deque.front()][max_front_match_index].size();i++) {
//				DMatch tmp_match = match_table[construct_deque.front()][max_front_match_index][i];
//				matched_keypoints_idx[construct_deque.front()].insert(tmp_match.queryIdx);
//				matched_keypoints_idx[max_front_match_index].insert(tmp_match.trainIdx);
//			}
//			cout << construct_deque.front() << " construct keypoints size:" << matched_keypoints_idx[construct_deque.front()].size() << endl;
//			cout << max_front_match_index << " construct keypoints size:" << matched_keypoints_idx[max_front_match_index].size() << endl;
//
//			//match_add 需要和上面的pair保持一致
//			vector<DMatch> match_add;
//			match_features(descriptor_for_all[construct_deque.front()], descriptor_for_all[max_front_match_index], match_add);
//			cout << "match size is " << match_add.size() << endl;
//			matches_for_all.push_back(match_add);
//
//			construct_deque.push_front(max_front_match_index);
//			for (vector<int>::iterator it = other_index_list.begin(); it != other_index_list.end();) {
//				if (*it == max_front_match_index) {
//					it = other_index_list.erase(it);
//				} else {
//					++it;
//				}
//			}
//
//		} else {
//			cout << "add image" << max_back_match_index << " to match " << construct_deque.back() << endl;
//			tmp_img_pair.clear();
//			tmp_img_pair.push_back(construct_deque.back());
//			tmp_img_pair.push_back(max_back_match_index);
//			img_add_order.push_back(tmp_img_pair);
//
//			//for new match method 
//			for (int i = 0; i < match_table[construct_deque.back()][max_back_match_index].size(); i++) {
//				DMatch tmp_match = match_table[construct_deque.back()][max_back_match_index][i];
//				matched_keypoints_idx[construct_deque.back()].insert(tmp_match.queryIdx);
//				matched_keypoints_idx[max_back_match_index].insert(tmp_match.trainIdx);
//			}
//			cout << construct_deque.back() << " construct keypoints size:" << matched_keypoints_idx[construct_deque.back()].size() << endl;
//			cout << max_back_match_index << " construct keypoints size:" << matched_keypoints_idx[max_back_match_index].size() << endl;
//
//
//			//match_add 需要和上面的pair保持一致
//			vector<DMatch> match_add;
//			match_features(descriptor_for_all[construct_deque.back()], descriptor_for_all[max_back_match_index], match_add);
//			cout << "match size is " << match_add.size() << endl;
//			matches_for_all.push_back(match_add);
//
//			construct_deque.push_back(max_back_match_index);
//			for (vector<int>::iterator it = other_index_list.begin(); it != other_index_list.end();) {
//				if (*it == max_back_match_index) {
//					it = other_index_list.erase(it);
//				} else {
//					++it;
//				}
//			}
//		}
//	}
//}


//旧的顺序添加照片match用于重建
//void match_features(vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all) {
//	matches_for_all.clear();
//	// n个图像，两两顺次有 n-1 对匹配
//	// 1与2匹配，2与3匹配，3与4匹配，以此类推
//	for (int i = 0; i < descriptor_for_all.size() - 1; ++i) {
//		cout << "Matching images " << i << " - " << i + 1;
//		vector<DMatch> matches;
//		match_features(descriptor_for_all[i], descriptor_for_all[i + 1], matches);
//		matches_for_all.push_back(matches);
//		cout << " :" << matches.size() << endl;
//	}
//}

void match_features(Mat& query, Mat& train, vector<DMatch>& matches) {
	vector<vector<DMatch>> knn_matches;
	//BFMatcher matcher(NORM_L2);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	matcher->knnMatch(query, train, knn_matches, 2);

	// 获取满足Ratio Test的最小匹配的距离
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r) {
		// Rotio Test
		if (knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance) {
			continue;
		}

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) {
			min_dist = dist;
		}
	}

	matches.clear();


	for (size_t r = 0; r < knn_matches.size(); ++r) {
		// 排除不满足Ratio Test的点和匹配距离过大的点
		if (
			knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			) {
			continue;
		}

		// 保存匹配点
		matches.push_back(knn_matches[r][0]);
	}
}

void match_features(Mat& query, Mat& train,
	vector<KeyPoint> query_keypoint, vector<KeyPoint> train_keypoint,
	vector<DMatch>& matches) {
	vector<vector<DMatch>> knn_matches;
	vector<DMatch> knn_good_matches;
	//BFMatcher matcher(NORM_L2);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	matcher->knnMatch(query, train, knn_matches, 2);

	// 获取满足Ratio Test 的最小匹配的距离
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r) {
		// Ratio Test
		if (knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance) {
			continue;
		}
		knn_good_matches.push_back(knn_matches[r][0]);

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) {
			min_dist = dist;
		}
	}
	matches.clear();

	//RANSAC 
	//如果匹配点太少，就不用ransac了，会有错误，直接匹配0
	vector<Point2f> p1, p2;
	get_matched_points(query_keypoint, train_keypoint, knn_good_matches, p1, p2);
	if (knn_good_matches.size()<8) {
		return;
	}
	vector<uchar> RansasStatus;
	Mat F_matrix = findFundamentalMat(p1, p2, RansasStatus, FM_RANSAC);
	if (RansasStatus.size()==0||RansasStatus.size()<knn_good_matches.size()) {
		//匹配点太少，少于8点了，无法计算F,所以这里的match应该为空
		//跳过
		return;
	} 
	for (int i = 0; i < knn_good_matches.size(); i++) {
		if (RansasStatus[i] != 0) {
			matches.push_back(knn_good_matches[i]);
		}
	}
	
	
}

void init_structure(
	Mat K,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<vector<Vec3b>>& colors_for_all,
	vector<vector<DMatch>>& matches_for_all,
	vector<vector<int>>& img_add_order,
	vector<Point3d>& structure,
	vector<vector<int>>& correspond_struct_idx,
	vector<Vec3b>& colors,
	vector<Mat>& rotations,
	vector<Mat>& motions) {

	//头两幅图像的下标
	int order_queryIdx = img_add_order[0][0], order_trainIdx = img_add_order[0][1];
	// 计算头两幅图像之间的变换矩阵
	vector<Point2f> p1, p2;
	vector<Vec3b> c2;
	Mat R, T;	// 旋转矩阵和平移向量
	Mat mask;	// mask中大于零的点代表匹配点，等于零的点代表失配点
	get_matched_points(key_points_for_all[order_queryIdx], key_points_for_all[order_trainIdx], matches_for_all[0], p1, p2);
	get_matched_colors(colors_for_all[order_queryIdx], colors_for_all[order_trainIdx], matches_for_all[0], colors, c2);

	//find_transform(K, p1, p2, R, T, mask);	// 三角分解得到R， T 矩阵
	if (!find_transform(K, p1, p2, R, T, mask)) {
		//引入的误差太大了，要跳过这张图片
		cout << "find transform ERROR!!!" << endl;
	}

	// 对头两幅图像进行三维重建
	maskout_points(p1, mask);
	maskout_points(p2, mask);
	maskout_colors(colors, mask);

	Mat R0 = Mat::eye(3, 3, CV_64FC1);
	Mat T0 = Mat::zeros(3, 1, CV_64FC1);
	reconstruct(K, R0, T0, R, T, p1, p2, structure);	// 三角化
														// 保存变换矩阵
	rotations.resize(key_points_for_all.size(), R0);
	motions.resize(key_points_for_all.size(), T0);
	rotations[order_queryIdx] = R0;
	rotations[order_trainIdx] = R;
	motions[order_queryIdx] = T0;
	motions[order_trainIdx] = T;
	//rotations = { R0, R };
	//motions = { T0, T };

	// 将correspond_struct_idx的大小初始化为与key_points_for_all完全一致
	correspond_struct_idx.clear();
	correspond_struct_idx.resize(key_points_for_all.size());
	for (int i = 0; i < key_points_for_all.size(); ++i) {
		correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);
	}

	// 填写头两幅图像的结构索引
	int idx = 0;
	vector<DMatch>& matches = matches_for_all[0];
	for (int i = 0; i < matches.size(); ++i) {
		if (mask.at<uchar>(i) == 0) {
			continue;
		}

		correspond_struct_idx[order_queryIdx][matches[i].queryIdx] = idx;	// 如果两个点对应的idx 相等 表明它们是同一特征点 idx 就是structure中对应的空间点坐标索引
		correspond_struct_idx[order_trainIdx][matches[i].trainIdx] = idx;
		++idx;
	}
}

void get_matched_points(
	vector<KeyPoint>& p1,
	vector<KeyPoint>& p2,
	vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2) {
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i) {
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		out_p2.push_back(p2[matches[i].trainIdx].pt);
	}
}

void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
) {
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i) {
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}

bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask) {
	// 根据内参数矩阵获取相机的焦距和光心坐标（主点坐标）
	double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	// 根据匹配点求取本征矩阵，使用RANSAC，进一步排除失配点
	Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
	if (E.empty()) {
		cout << "E is empty" << endl;
		return false;
	}

	double feasible_count = countNonZero(mask);	// 得到非零元素，即数组中的有效点
												// cout << (int)feasible_count << " - in - " << p1.size() << endl;

												// 对于RANSAC而言，outlier数量大于50%时，结果是不可靠的
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6) {
		cout << "outlier is > 50%" << endl;
		return false;
	}

	// 分解本征矩阵，获取相对变换
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	// cout << "pass_count = " << pass_count << endl;

	// 同时位于两个相机前方的点的数量要足够大
	if (((double)pass_count) / feasible_count < 0.7) {
		return false;
	}
	return true;
}

void maskout_points(vector<Point2f>& p1, Mat& mask) {
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i) {
		if (mask.at<uchar>(i) > 0) {
			p1.push_back(p1_copy[i]);
		}
	}
}

void maskout_colors(vector<Vec3b>& p1, Mat& mask) {
	vector<Vec3b> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i) {
		if (mask.at<uchar>(i) > 0) {
			p1.push_back(p1_copy[i]);
		}
	}
}

void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3d>& structure) {
	// 两个相机的投影矩阵[R, T], triangulatePoints只支持float型
	Mat proj1(3, 4, CV_32FC1);
	Mat proj2(3, 4, CV_32FC1);

	R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);
	//T1.convertTo(proj2(Range(0, 3), Range(3, 4)), CV_32FC1);
	T1.convertTo(proj1.col(3), CV_32FC1);

	R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	//T2.convertTo(proj2(Range(0, 3), Range(3, 4)), CV_32FC1);
	T2.convertTo(proj2.col(3), CV_32FC1);

	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK * proj1;
	proj2 = fK * proj2;

	// 三角重建
	Mat s;
	triangulatePoints(proj1, proj2, p1, p2, s);

	structure.clear();
	structure.reserve(s.cols);
	for (int i = 0; i < s.cols; ++i) {
		Mat_<float> col = s.col(i);
		col /= col(3);	// 齐次坐标
		structure.push_back(Point3d(col(0), col(1), col(2)));
	}
}

void get_objpoints_and_imgpoints(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<Point3d>& structure,
	vector<KeyPoint>& key_points,
	vector<Point3d>& object_points,
	vector<Point2f>& image_points) {
	object_points.clear();
	image_points.clear();

	for (int i = 0; i < matches.size(); ++i) {
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];
		if (struct_idx < 0)	// 表明跟前一副图像没有匹配点
		{
			continue;
		}

		object_points.push_back(structure[struct_idx]);
		image_points.push_back(key_points[train_idx].pt);	// train中对应关键点的坐标 二维
	}
}

void fusion_structure(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<Point3d>& structure,
	vector<Point3d>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors) {
	for (int i = 0; i < matches.size(); ++i) {
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];
		if (struct_idx >= 0)	// 若该点在空间中已经存在，则这对匹配点对应的空间点应该是同一个，索引要相同
		{
			next_struct_indices[train_idx] = struct_idx;
			continue;
		}

		// 若该点在空间中未存在，将该点加入到结构中，且这对匹配点的空间索引都为新加入的点的索引
		structure.push_back(next_structure[i]);
		colors.push_back(next_colors[i]);
		struct_indices[query_idx] = next_struct_indices[train_idx] = structure.size() - 1;
	}
}

void save_ply_file(string file_name, vector<Point3d> &structure_points, vector<Vec3b> &color) {
	ofstream ply_file;
	ply_file.open(file_name + ".ply");
	ply_file << "ply" << endl;
	ply_file << "format ascii 1.0 " << endl;
	ply_file << "element vertex " << structure_points.size() << endl;
	ply_file << "property float32 x " << endl
		<< "property float32 y " << endl
		<< "property float32 z" << endl
		<< "property uchar red  " << endl
		<< "property uchar green" << endl
		<< "property uchar blue" << endl
		<< "element face 0" << endl
		<< "property list uchar int vertex_indices" << endl
		<< "end_header" << endl;
	for (size_t i = 0; i < structure_points.size(); i++) {
		if (i < color.size()) {
			ply_file << structure_points[i].x << " "
				<< structure_points[i].y << " "
				<< structure_points[i].z << " "
				<< (int)color[i].val[2] << " "
				<< (int)color[i].val[1] << " "
				<< (int)color[i].val[0] << endl;
		} else {
			ply_file << structure_points[i].x << " "
				<< structure_points[i].y << " "
				<< structure_points[i].z << " "
				<< "0 0 0" << endl;
		}
	}
	ply_file.close();
}

//获取特定格式的文件名    
//void getAllFiles(string path, vector<string>& files, string format)
//{
//	long  hFile = 0;//文件句柄  64位下long 改为 intptr_t
//	struct _finddata_t fileinfo;//文件信息 
//	string p;
//	if ((hFile = _findfirst(p.assign(path).append("\\*" + format).c_str(), &fileinfo)) != -1) //文件存在
//	{
//		do
//		{
//			if ((fileinfo.attrib & _A_SUBDIR))//判断是否为文件夹
//			{
//				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)//文件夹名中不含"."和".."
//				{
//					files.push_back(p.assign(path).append("\\").append(fileinfo.name)); //保存文件夹名
//					getAllFiles(p.assign(path).append("\\").append(fileinfo.name), files, format); //递归遍历文件夹
//				}
//			}
//			else
//			{
//				files.push_back(p.assign(path).append("\\").append(fileinfo.name));//如果不是文件夹，储存文件名
//			}
//		} while (_findnext(hFile, &fileinfo) == 0);
//		_findclose(hFile);
//	}
//}

void bundle_adjustment(
	Mat& intrinsic,
	vector<Mat>& extrinsics,
	vector<vector<int>>& correspond_struct_idx,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Point3d>& structure) {

	ceres::Problem problem;

	// load extrinsics (rotations and motions)
	for (size_t i = 0; i < extrinsics.size(); ++i) {
		problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);
	}
	// fix the first camera.
	problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());

	// load intrinsic
	problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy
	 // load points
	ceres::LossFunction* loss_function = new ceres::HuberLoss(4);   // loss function make bundle adjustment robuster.
	for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx) {
		vector<int>& point3d_ids = correspond_struct_idx[img_idx];
		vector<KeyPoint>& key_points = key_points_for_all[img_idx];
		for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx) {
			int point3d_id = point3d_ids[point_idx];
			if (point3d_id < 0)
				continue;

			//after remove error points 后会报错
			if (point3d_id>=structure.size()) {
				continue;
			}
			Point2d observed = key_points[point_idx].pt;
			// 模板参数中，第一个为代价函数的类型，第二个为代价的维度，剩下三个分别为代价函数第一第二还有第三个参数的维度
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));

			problem.AddResidualBlock(
				cost_function,
				loss_function,
				intrinsic.ptr<double>(),            // Intrinsic
				extrinsics[img_idx].ptr<double>(),  // View Rotation and Translation
				&(structure[point3d_id].x)          // Point in 3D space
			);
		}
	}
	cout << "before solve BA" << endl;

	// Solve BA
	ceres::Solver::Options ceres_config_options;
	ceres_config_options.minimizer_progress_to_stdout = false;
	ceres_config_options.logging_type = ceres::SILENT;
	ceres_config_options.num_threads = 1;
	ceres_config_options.preconditioner_type = ceres::SCHUR_JACOBI;
	ceres_config_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
	ceres_config_options.use_inner_iterations = true;
	ceres_config_options.max_num_iterations = 100;
	ceres_config_options.minimizer_progress_to_stdout = true;
	//ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

	ceres::Solver::Summary summary;
	ceres::Solve(ceres_config_options, &problem, &summary);

	if (!summary.IsSolutionUsable()) {
		cout << "Bundle Adjustment failed." << endl;
	} else {
		// Display statistics about the minimization
		cout << "Bundle Adjustment statistics (approximated RMSE):\n"
			<< " #views: " << extrinsics.size() << "\n"
			<< " #residuals: " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
			<< " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
			<< " Time (s): " << summary.total_time_in_seconds << "\n"
			<< endl;
	}
}


void bundle_adjustment(
	vector<int> reconstred_image_idx,
	Mat& intrinsic,
	vector<Mat>& extrinsics,
	vector<vector<int>>& correspond_struct_idx,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Point3d>& structure) {

	ceres::Problem problem;

	// load extrinsics (rotations and motions)
	for (size_t i = 0; i < extrinsics.size(); ++i) {
		problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);
	}
	// fix the first camera.
	problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());

	// load intrinsic
	problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy
   // load points
	ceres::LossFunction* loss_function = new ceres::HuberLoss(4);   // loss function make bundle adjustment robuster.
	for (size_t img_idx_idx = 0; img_idx_idx < reconstred_image_idx.size(); ++img_idx_idx) {

		size_t img_idx = reconstred_image_idx[img_idx_idx];
		//这里img_idx表达的是 照片的下标，reconstructed_img_idx可以表示
		vector<int>& point3d_ids = correspond_struct_idx[img_idx];
		vector<KeyPoint>& key_points = key_points_for_all[img_idx];
		for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx) {
			int point3d_id = point3d_ids[point_idx];
			if (point3d_id < 0)
				continue;

			//after remove error points 后会报错
			if (point3d_id >= structure.size()) {
				continue;
			}
			Point2d observed = key_points[point_idx].pt;
			// 模板参数中，第一个为代价函数的类型，第二个为代价的维度，剩下三个分别为代价函数第一第二还有第三个参数的维度
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));

			problem.AddResidualBlock(
				cost_function,
				loss_function,
				intrinsic.ptr<double>(),            // Intrinsic
				extrinsics[img_idx_idx].ptr<double>(),  // View Rotation and Translation
				&(structure[point3d_id].x)          // Point in 3D space
			);
		}
	}
	cout << "before solve BA" << endl;

	// Solve BA
	ceres::Solver::Options ceres_config_options;
	ceres_config_options.minimizer_progress_to_stdout = false;
	ceres_config_options.logging_type = ceres::SILENT;
	ceres_config_options.num_threads = 1;
	ceres_config_options.preconditioner_type = ceres::SCHUR_JACOBI;
	ceres_config_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
	ceres_config_options.use_inner_iterations = true;
	ceres_config_options.max_num_iterations = 100;
	ceres_config_options.minimizer_progress_to_stdout = true;
	//ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

	ceres::Solver::Summary summary;
	ceres::Solve(ceres_config_options, &problem, &summary);

	if (!summary.IsSolutionUsable()) {
		cout << "Bundle Adjustment failed." << endl;
	} else {
		// Display statistics about the minimization
		cout << "Bundle Adjustment statistics (approximated RMSE):\n"
			<< " #views: " << extrinsics.size() << "\n"
			<< " #residuals: " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
			<< " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
			<< " Time (s): " << summary.total_time_in_seconds << "\n"
			<< endl;
	}
}

/**
* NVM format
NVM V3

<camera_numbers>
image_name estimated_focal_length QW QX QY QZ Cx Cy Cz D0 0
...
...
...

<point_numbers>
X Y Z R G B <tracks_numbers>: image_index feature_index x y
*/
void save_nvm_file(string file_name, 
	vector<string> img_names,
	Mat K,
	vector<Mat> rorations, 
	vector<Mat> motions, 
	vector<Point3d> &structure_points,
	vector<Vec3b> &color,
	vector<vector<KeyPoint>> key_points_for_all, 
	vector<vector<int>> correspond_struct_idx) {
	ofstream nvm_file;
	nvm_file.open(file_name + ".nvm");
	nvm_file << "NVM_V3" << endl << endl;
	//camera
	nvm_file << img_names.size() << endl;
	for (int i = 0; i < img_names.size();i++) {
		nvm_file << img_names[i] << " ";
		double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
		nvm_file << focal_length << " ";
		Eigen::Matrix3d R_n;
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> T_n;
		cv2eigen(rorations[i], R_n);
		cv2eigen(motions[i], T_n);

		Eigen::Quaterniond q(R_n);
		nvm_file << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " ";
		//nvm_file << q.x() << " " << -1.0*q.w() << " " << q.z() << " " << -1.0*q.y() << " ";

		Eigen::Vector3d p_pose;
		p_pose = -R_n.inverse()*T_n;
		nvm_file << p_pose.x() << " " << p_pose.y() << " " << p_pose.z() << " ";

		//D0
		nvm_file << "0" <<" ";

		nvm_file << "0 " << endl;
	}
	nvm_file << endl;

	//points
	double image_width = K.at<double>(2);
	double image_height = K.at<double>(5);
	vector<vector<float>> track_list;
	//track_num {img_index kp_index kp_ptx kp_pty}
	for (int s_index = 0; s_index < structure_points.size(); s_index++) {
		vector<float> track_tmp;
		track_tmp.clear();
		for (int img_index = 0; img_index < key_points_for_all.size(); img_index++) {
			for (int f_index = 0; f_index < key_points_for_all[img_index].size(); f_index++) {
				if (correspond_struct_idx[img_index][f_index] == s_index) {
					//cout << img_index << f_index << key_points_for_all[img_index][f_index].pt << endl;
					track_tmp.push_back(img_index);
					track_tmp.push_back(f_index);
					track_tmp.push_back(key_points_for_all[img_index][f_index].pt.x-image_width);
					track_tmp.push_back(key_points_for_all[img_index][f_index].pt.y-image_height);
				}
			}
		}
		track_list.push_back(track_tmp);
	}
	nvm_file << structure_points.size() << endl;
	for (int i = 0; i < structure_points.size();i++) {
		if (i < color.size()) {
			nvm_file << structure_points[i].x << " "
				<< structure_points[i].y << " "
				<< structure_points[i].z << " "
				<< (int)color[i].val[2] << " "
				<< (int)color[i].val[1] << " "
				<< (int)color[i].val[0] << " ";
		} else {
			nvm_file << structure_points[i].x << " "
				<< structure_points[i].y << " "
				<< structure_points[i].z << " "
				<< "0 0 0" << " ";
		}
		nvm_file << track_list[i].size() / 4 << " ";
		for (int j = 0; j < track_list[i].size(); j++) {
			nvm_file << track_list[i][j] << " ";
		}
		nvm_file << endl;
	}
	nvm_file << endl;

	//ply comment
	nvm_file << "0 " << endl;
	nvm_file << "#the last part of NVM file points to the PLY files" << endl
		<< "#the first number is the number of associated PLY files" << endl
		<< "#each following number gives a model-index that has PLY" << endl
		<< "0 " << endl;

	nvm_file.close();
}

void save_camera_positions(vector<Mat> rorations, vector<Mat> motions) {
	ofstream camera_file;
	camera_file.open("camera.txt");

	for (int i = 0; i < rorations.size();i++) {
		int r_height = rorations[i].rows;
		int r_width = rorations[i].cols;
		int t_height = motions[i].rows;
		int t_width = motions[i].cols;
		camera_file << i << endl;
		for (int h = 0; h < r_height;h++) {
			for (int w = 0; w < r_width;w++) {
				//camera_file << (float)rorations[i].ptr<uchar>(h)[w] << '\t';
				camera_file << rorations[i].at<double>(h, w) << " ";
			}
			camera_file << endl;
		}
		for (int h = 0; h < t_height; h++) {
			for (int w = 0; w < t_width; w++) {
				//camera_file << (float)motions[i].ptr<uchar>(h)[w] << '\t';
				camera_file << motions[i].at<double>(h, w) << " ";
			}
		}
		camera_file << endl;

		
	}
	camera_file.close();
}

void save_tracks(vector<Point3d> structure, vector<vector<KeyPoint>> key_points_for_all, vector<vector<int>> correspond_struct_idx) {
	ofstream track_file;
	track_file.open("tracks.txt");
	vector<vector<float>> track_list;
	//track_num {img_index kp_index kp_ptx kp_pty}
	for (int s_index = 0; s_index < structure.size();s_index++) {
		vector<float> track_tmp;
		track_tmp.clear();
		for (int img_index = 0; img_index < key_points_for_all.size();img_index++) {
			for (int f_index = 0; f_index < key_points_for_all[img_index].size();f_index++) {
				if (correspond_struct_idx[img_index][f_index]==s_index) {
					//cout << img_index << f_index << key_points_for_all[img_index][f_index].pt << endl;
					track_tmp.push_back(img_index);
					track_tmp.push_back(f_index);
					track_tmp.push_back(key_points_for_all[img_index][f_index].pt.x);
					track_tmp.push_back(key_points_for_all[img_index][f_index].pt.y);
				}
			}
		}
		track_list.push_back(track_tmp);
	}
	cout << "track list size is " << track_list.size() << endl;
	for (int i = 0; i < track_list.size();i++) {
		track_file << track_list[i].size()/4 << " ";
		for (int j = 0; j < track_list[i].size();j++) {
			track_file << track_list[i][j] << " ";
		}
		track_file << endl;
	}
	track_file.close();
}

void save_bundle_file(string file_name,
	Mat K,
	vector<Mat>& rotations,
	vector<Mat>& motions,
	vector<Point3d>& structure,
	vector<Vec3b>& colors,
	vector<vector<KeyPoint>> key_points_for_all,
	vector<vector<int>> correspond_struct_idx) {
	ofstream bundle_file;
	bundle_file.open(file_name + ".out");
	bundle_file << "# Bundle file v0.3" << endl;
	bundle_file << rotations.size() << " " << structure.size() << endl;
	//camera
	double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
	for (int i = 0; i < rotations.size();i++) {
		bundle_file << focal_length << " 0 0 " << endl;
		int r_height = rotations[i].rows;
		int r_width = rotations[i].cols;
		int t_height = motions[i].rows;
		int t_width = motions[i].cols;
		for (int h = 0; h < r_height; h++) {
			for (int w = 0; w < r_width; w++) {
				bundle_file << rotations[i].at<double>(h, w) << " ";
			}
			bundle_file << endl;
		}
		for (int h = 0; h < t_height; h++) {
			for (int w = 0; w < t_width; w++) {
				bundle_file << motions[i].at<double>(h, w) << " ";
			}
		}
		bundle_file << endl;
	}
	//points
	double image_width = K.at<double>(2);
	double image_height = K.at<double>(5);
	vector<vector<double>> track_list;
	//track_num {img_index kp_index kp_ptx kp_pty}
	for (int s_index = 0; s_index < structure.size(); s_index++) {
		vector<double> track_tmp;
		track_tmp.clear();
		for (int img_index = 0; img_index < key_points_for_all.size(); img_index++) {
			for (int f_index = 0; f_index < key_points_for_all[img_index].size(); f_index++) {
				if (correspond_struct_idx[img_index][f_index] == s_index) {
					//cout << img_index << f_index << key_points_for_all[img_index][f_index].pt << endl;
					track_tmp.push_back(img_index);
					track_tmp.push_back(f_index);
					track_tmp.push_back(key_points_for_all[img_index][f_index].pt.x - image_width);
					track_tmp.push_back(key_points_for_all[img_index][f_index].pt.y - image_height);
				}
			}
		}
		track_list.push_back(track_tmp);
	}
	for (int i = 0; i < structure.size(); i++) {
		if (i < colors.size()) {
			bundle_file << structure[i].x << " "
				<< structure[i].y << " "
				<< structure[i].z << " "<<endl
				<< (int)colors[i].val[2] << " "
				<< (int)colors[i].val[1] << " "
				<< (int)colors[i].val[0] << " "<<endl;
		} else {
			bundle_file << structure[i].x << " "
				<< structure[i].y << " "
				<< structure[i].z << " "<<endl
				<< "0 0 0" << " "<<endl;
		}
		//view list
		bundle_file << track_list[i].size() / 4 << " ";
		for (int j = 0; j < track_list[i].size(); j++) {
			bundle_file << track_list[i][j] << " ";
		}
		bundle_file << endl;
	}

}

void test() {
	Mat R(Matx33d(
		0.376734172473, -0.0107271861894, 0.926259322542,
		0.164921846532, -0.983180986382, -0.0784644939512,
		0.911522031022, 0.182320617757, -0.36862870578));
	Mat t(Matx31d(-0.92293710287, -0.0195358013302, -0.252308258544));

	Eigen::Matrix3d R_n;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> T_n;
	cv2eigen(R, R_n);
	cv2eigen(t, T_n);

	Eigen::Quaterniond q(R_n);
	cout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "<<endl;
	//cout << q.coeffs() << endl;

	Eigen::Vector3d p_pose;
	p_pose = -R_n.inverse()*T_n;
	cout << p_pose.x() << " " << p_pose.y() << " " << p_pose.z() << " "<<endl;
}

//把误差过大的点删掉，主要是偏移太大的点
void remove_error_points(vector<Point3d>& structure) {
	for (vector<Point3d>::iterator it = structure.begin(); it != structure.end();) {
		//空间坐标太大或者太小的，先设置阈值为100
		if (abs(it->x)>=100||abs(it->y)>=100||abs(it->z)>=100) {
			it = structure.erase(it);
		} else {
			++it;
		}
	}
}

string get_filename_from_path(string path) {
	char szDrive[_MAX_DRIVE];   //磁盘名
	char szDir[_MAX_DIR];       //路径名
	char szFname[_MAX_FNAME];   //文件名
	char szExt[_MAX_EXT];       //后缀名
	_splitpath(path.c_str(), szDrive, szDir, szFname, szExt); //分解路径
	return string(szFname) + string(szExt);
}

Mat rorate90(Mat src) {

	Mat src_copy = Mat(src.rows, src.cols, src.depth());
	transpose(src, src_copy);
	flip(src_copy, src_copy, 0);
	return src_copy;
}