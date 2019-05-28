#pragma once

#include <vector>
#include <io.h>
#include <fstream>
#include <string>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
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

//重投影误差
struct ReprojectCost {
	cv::Point2d observation;

	ReprojectCost(cv::Point2d& observation)
		: observation(observation) {
	}

	template <typename T>
	bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const {
		const T* r = extrinsic;
		const T* t = &extrinsic[3];

		T pos_proj[3];
		ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);

		// Apply the camera translation
		pos_proj[0] += t[0];
		pos_proj[1] += t[1];
		pos_proj[2] += t[2];

		const T x = pos_proj[0] / pos_proj[2];
		const T y = pos_proj[1] / pos_proj[2];

		const T fx = intrinsic[0];
		const T fy = intrinsic[1];
		const T cx = intrinsic[2];
		const T cy = intrinsic[3];

		// Apply intrinsic
		const T u = fx * x + cx;
		const T v = fy * y + cy;

		residuals[0] = u - T(observation.x);
		residuals[1] = v - T(observation.y);

		return true;
	}
};

//关键点转为照片像素坐标
void get_matched_points(
	vector<KeyPoint>& p1,
	vector<KeyPoint>& p2,
	vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2);
void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
);

//利用K和照片像素坐标恢复相机位姿R&t
bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask);

//通过mask去掉误差匹配的点和颜色
void maskout_points(vector<Point2f>& p1, Mat& mask);
void maskout_colors(vector<Vec3b>& p1, Mat& mask);

//利用K&相机位姿&照片像素坐标恢复3D空间点
void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, 
	vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3d>& structure);

//
void get_objpoints_and_imgpoints(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<Point3d>& structure,
	vector<KeyPoint>& key_points,
	vector<Point3d>& object_points,
	vector<Point2f>& image_points);

//点云融合
void fusion_structure(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<Point3d>& structure,
	vector<Point3d>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors);

//提取特征
void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector <vector<Vec3b>>& colors_for_all
);

//匹配两张照片---KNN
void match_features(Mat& query, Mat& train, vector<DMatch>& matches);

//匹配两张照片---KNN+RANSAC
void match_features(Mat& query, Mat& train,
	vector<KeyPoint> query_keypoint, vector<KeyPoint> train_keypoint,
	vector<DMatch>& matches);

//匹配所有照片，并根据匹配数量决定添加照片重建的顺序
void match_features(vector<Mat>& descriptor_for_all, 
	vector<vector<DMatch>>& matches_for_all, 
	vector<vector<int>>& img_add_order);

//匹配所有照片，并根据匹配数量决定添加照片重建的顺序
void match_features(vector<Mat>& descriptor_for_all,
	vector<vector<KeyPoint>> keyPoints_for_all,
	vector<DMatch> match_table[100][100],
	vector<vector<DMatch>>& matches_for_all,
	vector<vector<int>>& img_add_order);

//多目重建：初始重建第一对照片
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
	vector<Mat>& motions
);

//BA优化
void bundle_adjustment(
	Mat& intrinsic,
	vector<Mat>& extrinsics,
	vector<vector<int>>& correspond_struct_idx,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Point3d>& structure);

//因为参数都是提前resize好的，只有部分照片重建时，需要添加现在已重建的照片list,然后取值去算
void bundle_adjustment(
	vector<int> reconstred_image_idx,
	Mat& intrinsic,
	vector<Mat>& extrinsics,
	vector<vector<int>>& correspond_struct_idx,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Point3d>& structure);

//保存3D空间点
void save_structure(string file_name,
	vector<Mat>& rotations,
	vector<Mat>& motions,
	vector<Point3d>& structure, vector<Vec3b>& colors);

void save_ply_file(string file_name, vector<Point3d> &structure_points, vector<Vec3b> &color);

void save_bundle_file(string file_name,
	Mat K,
	vector<Mat>& rotations,
	vector<Mat>& motions,
	vector<Point3d>& structure,
	vector<Vec3b>& colors,
	vector<vector<KeyPoint>> key_points_for_all,
	vector<vector<int>> correspond_struct_idx);

void save_nvm_file(string file_name,
	vector<string> img_names,
	Mat K,
	vector<Mat> rorations,
	vector<Mat> motions,
	vector<Point3d> &structure_points,
	vector<Vec3b> &color,
	vector<vector<KeyPoint>> key_points_for_all,
	vector<vector<int>> correspond_struct_idx);

void save_camera_positions(vector<Mat> rorations, vector<Mat> motions);

void save_tracks(vector<Point3d> structure, vector<vector<KeyPoint>> key_points_for_all, vector<vector<int>> correspond_struct_idx);

void test();

string get_filename_from_path(string path);

void remove_error_points(vector<Point3d>& structure);

Mat rorate90(Mat src);
