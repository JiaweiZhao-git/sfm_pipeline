#include <iostream>
#include "sfm.h"

using namespace std;
using namespace cv;

//C:\Users\15297\Desktop\OpenCV_SFM\image
string dir = "C:\\Users\\15297\\Desktop\\OpenCV_SFM\\image\\elephant_test\\";

int main(int argc, char** argv)
{
	string data_name = "elephant_test";
	vector<string> img_names;
	glob(dir, img_names, false);
	
	// 本征矩阵
	/*Mat K(Matx33d(
		2759.48, 0, 1520.69,
		0, 2764.16, 1006.81,
		0, 0, 1));*/
		
	//for qhm
	/*Mat K(Matx33d(1708.156424581, 0, 1092,
		0, 1708.156424581, 728,
		0, 0, 1));*/

	//for ET
	/*Mat K(Matx33d(600.375234521, 0, 320,
		0, 600.375234521, 240,
		0, 0, 1));*/

	//for statue
	/*Mat K(Matx33d(3410.99, 0, 3129.34,
		0, 3412.37, 2059.72,
		0, 0, 1));*/

	//for elephant
	Mat K(Matx33d(4905.9829059829, 0, 1148,
		0, 4905.9829059829, 764,
		0, 0, 1));

	//for cathedral
	/*Mat K(Matx33d(2955.1304347826, 0, 1416,
		0, 2955.1304347826, 1064,
		0, 0, 1));*/

	vector<vector<KeyPoint>> key_points_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matches_for_all;
	// 提取所有图像的特征
	extract_features(img_names, key_points_for_all, descriptor_for_all, colors_for_all);

	// 对所有图像进行顺次的特征匹配
	vector<vector<int>> img_add_order;//添加照片的顺序，如[[6,7],[6,5],[7,8],[5,4]]
	int init_idx1, init_idx2;
	vector<DMatch> match_table[100][100];
	match_features(descriptor_for_all, key_points_for_all,match_table,init_idx1,init_idx2);

	/*cv::Mat img1;
	cv::Mat img2;
	for (int i = 0; i < img_add_order.size(); i++) {
		int query_idx = img_add_order[i][0];
		int train_idx = img_add_order[i][1];
		img1 = cv::imread(img_names[query_idx]);
		img2 = cv::imread(img_names[train_idx]);
		cv::Mat out;
		drawMatches(img1, key_points_for_all[query_idx], img2, key_points_for_all[train_idx], matches_for_all[i], out);
		cv::imwrite("F:\\DATASET\\qinghuamen\\qhm-small-tnt\\fountain_matches\\" + string("/") + to_string(query_idx)
			+ string("-") + to_string(train_idx) + string(".jpg"), out);
	}*/

	vector<Point3d> structure;
	vector<vector<int>> correspond_struct_idx;	// 保存第i副图像中第j特征点对应的structure中点的索引
	vector<Vec3b> colors;
	vector<Mat> rotations;//注意：下标对应照片，所以img6是rotations[6]，所以需要先初始化大小
	vector<Mat> motions;//同上
	vector<int> reconstructed_image_idx;//刚开始为空，每次增量重建的添加照片id进去，作为BA参数
	set<int> remain_image_idx;//刚开始为满，每次删除已重建的id

	for (int i = 0; i < img_names.size();i++) {
		remain_image_idx.insert(i);
	}

	// 初始化结构（三维点云） 头两幅
	init_structure(
		K,
		key_points_for_all,
		colors_for_all,
		match_table[init_idx1][init_idx2],
		init_idx1,
		init_idx2,
		structure,
		correspond_struct_idx,
		colors,
		rotations,
		motions
	);


	//第一次BA:初始化重建后
	reconstructed_image_idx.push_back(init_idx1);
	reconstructed_image_idx.push_back(init_idx2);
	remain_image_idx.erase(init_idx1);
	remain_image_idx.erase(init_idx2);
	Mat intrinsic_init(Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
	vector<Mat> extrinsics_init;
	for (size_t i = 0; i < reconstructed_image_idx.size(); ++i) {
		Mat extrinsic_init(6, 1, CV_64FC1);
		Mat r;
		Rodrigues(rotations[reconstructed_image_idx[i]], r);

		r.copyTo(extrinsic_init.rowRange(0, 3));
		motions[reconstructed_image_idx[i]].copyTo(extrinsic_init.rowRange(3, 6));

		extrinsics_init.push_back(extrinsic_init);
	}

	std::cout << "before init BA 3D size is:" << structure.size() << endl;

	bundle_adjustment(reconstructed_image_idx, intrinsic_init, extrinsics_init, correspond_struct_idx, key_points_for_all, structure);
	cout << "K: " << K << " ---> ";
	K.at<double>(0, 0) = intrinsic_init.at<double>(0, 0);
	K.at<double>(1, 1) = intrinsic_init.at<double>(1, 0);
	K.at<double>(0, 2) = intrinsic_init.at<double>(2, 0);
	K.at<double>(1, 2) = intrinsic_init.at<double>(3, 0);
	cout << K << endl;
	for (size_t i = 0; i < reconstructed_image_idx.size(); ++i) {
		Mat extrinsic = extrinsics_init[i];
		Mat r;
		cout << "image " << reconstructed_image_idx[i] << endl;
		cout << "R: " << rotations[reconstructed_image_idx[i]] << " ---> ";
		extrinsic.rowRange(0, 3).copyTo(r);
		Rodrigues(r, rotations[reconstructed_image_idx[i]]);
		cout << rotations[reconstructed_image_idx[i]] << endl;
		cout << "T: " << motions[reconstructed_image_idx[i]] << " ---> ";
		extrinsic.rowRange(3, 6).copyTo(motions[reconstructed_image_idx[i]]);
		cout << motions[reconstructed_image_idx[i]] << endl;
	}

	/*remove_error_points(structure);
	cout << "after remove error points 3D size is:" << structure.size() << endl;*/
	save_ply_file(data_name+"_init", structure, colors);


	while (!remain_image_idx.empty()) {
		//add points by seeing how many used 
		int max_see = 1;
		int max_idx;
		for (set<int>::iterator it = remain_image_idx.begin(); it != remain_image_idx.end(); it++) {
			int tmp_see = 0;

			//every camera sees how many
			for (int i = 0; i < reconstructed_image_idx.size(); i++) {
				int tmp_idx = reconstructed_image_idx[i];
				vector<DMatch> tmp_match_list = match_table[*it][tmp_idx];
				//cout << *it << " match " << tmp_idx << " size " << tmp_match_list.size() << endl;
				for (int m = 0; m < tmp_match_list.size(); m++) {
					if (correspond_struct_idx[tmp_idx][tmp_match_list[m].trainIdx] > 0) {
						tmp_see++;
					}

				}
			}
			cout << *it << " sees " << tmp_see << " points" << endl;
			if (max_see < tmp_see) {
				max_see = tmp_see;
				max_idx = *it;
			}
		}

		cout << "register image " << max_idx << endl;
		remain_image_idx.erase(max_idx);
		reconstructed_image_idx.push_back(max_idx);

		int max_match_size = 1;
		for (int i = 0; i < reconstructed_image_idx.size() - 1; i++) {
			int order_queryIdx = reconstructed_image_idx[i], order_trainIdx = max_idx;
			
			if (max_match_size< match_table[order_queryIdx][order_trainIdx].size()) {
				max_match_size = match_table[order_queryIdx][order_trainIdx].size();
			}
		}

		for (int i = 0; i < reconstructed_image_idx.size()-1; i++) {
			int order_queryIdx = reconstructed_image_idx[i], order_trainIdx = max_idx;
			if (match_table[order_queryIdx][order_trainIdx].size()<0.75*max_match_size) {
				continue;
			}
			cout << "add match " << order_queryIdx << " --- " << order_trainIdx << " size:";
			cout << match_table[order_queryIdx][order_trainIdx].size() << endl;
			vector<Point3d> object_points;
			vector<Point2f> image_points;
			Mat r, R, T;

			// 获取第order_queryIdx副图像中匹配点对应的三维点，以及在第order_trainIdx副图像中对应的像素点
			get_objpoints_and_imgpoints(
				match_table[order_queryIdx][order_trainIdx],
				correspond_struct_idx[order_queryIdx],
				structure,
				key_points_for_all[order_trainIdx],
				object_points,
				image_points
			);

			// 求解变换矩阵
			if (object_points.size()<4||image_points.size()<4) {
				continue;
			}
			bool pp = solvePnPRansac(object_points, image_points, K, noArray(), r, T);
			cout << "solvePnPRansac is " << pp << endl;
			if (!pp) {
				continue;
			}
			// 将旋转向量转换为旋转矩阵
			Rodrigues(r, R);
			// 保存变换矩阵
			rotations[order_trainIdx] = R;
			motions[order_trainIdx] = T;
			//rotations.push_back(R);
			//motions.push_back(T);

			vector<Point2f> p1, p2;
			vector<Vec3b> c1, c2;
			get_matched_points(key_points_for_all[order_queryIdx], key_points_for_all[order_trainIdx], match_table[order_queryIdx][order_trainIdx], p1, p2);
			get_matched_colors(colors_for_all[order_queryIdx], colors_for_all[order_trainIdx], match_table[order_queryIdx][order_trainIdx], c1, c2);

			// 根据之前求得的R, T进行三维重建
			vector<Point3d> next_structure;
			reconstruct(K, rotations[order_queryIdx], motions[order_queryIdx], R, T, p1, p2, next_structure);

			//将新的重建结果与之前的融合
			fusion_structure(
				match_table[order_queryIdx][order_trainIdx],
				correspond_struct_idx[order_queryIdx],
				correspond_struct_idx[order_trainIdx],
				structure,
				next_structure,
				colors,
				c1
			);
			cout << "after fusion structure size is " << structure.size() << endl;
		}


		if (reconstructed_image_idx.size()%2==0) {
			Mat intrinsic_add(Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
			vector<Mat> extrinsics_add;
			for (size_t i = 0; i < reconstructed_image_idx.size(); ++i) {
				Mat extrinsic_add(6, 1, CV_64FC1);
				Mat r;
				Rodrigues(rotations[reconstructed_image_idx[i]], r);

				r.copyTo(extrinsic_add.rowRange(0, 3));
				motions[reconstructed_image_idx[i]].copyTo(extrinsic_add.rowRange(3, 6));

				extrinsics_add.push_back(extrinsic_add);
			}

			std::cout << "before add BA 3D size is:" << structure.size() << endl;

			bundle_adjustment(reconstructed_image_idx, intrinsic_add, extrinsics_add, correspond_struct_idx, key_points_for_all, structure);
			cout << "K: " << K << " ---> ";
			K.at<double>(0, 0) = intrinsic_add.at<double>(0, 0);
			K.at<double>(1, 1) = intrinsic_add.at<double>(1, 0);
			K.at<double>(0, 2) = intrinsic_add.at<double>(2, 0);
			K.at<double>(1, 2) = intrinsic_add.at<double>(3, 0);
			cout << K << endl;

			for (size_t i = 0; i < reconstructed_image_idx.size(); ++i) {
				Mat extrinsic = extrinsics_add[i];
				Mat r;
				cout << "image " << reconstructed_image_idx[i] << endl;
				cout << "R: " << rotations[reconstructed_image_idx[i]] << " ---> ";
				extrinsic.rowRange(0, 3).copyTo(r);
				Rodrigues(r, rotations[reconstructed_image_idx[i]]);
				cout << rotations[reconstructed_image_idx[i]] << endl;
				cout << "T: " << motions[reconstructed_image_idx[i]] << " ---> ";
				extrinsic.rowRange(3, 6).copyTo(motions[reconstructed_image_idx[i]]);
				cout << motions[reconstructed_image_idx[i]] << endl;
			}
			//remove_error_points(structure, correspond_struct_idx);
			int tmp_max = 0;
			for (int i = 0; i < correspond_struct_idx.size(); i++) {
				for (int j = 0; j < key_points_for_all[i].size(); j++) {
					if (correspond_struct_idx[i][j] > tmp_max) {
						tmp_max = correspond_struct_idx[i][j];
					}
				}
			}
			cout << "correspond struct idx max is :" << tmp_max << endl;
			cout << "structure size is " << structure.size() << endl;
			//bundle_adjustment(reconstructed_image_idx, intrinsic_add, extrinsics_add, correspond_struct_idx, key_points_for_all, structure);
		}
		
		save_ply_file(data_name + "add" + to_string(max_idx), structure, colors);
		save_ply_file(data_name + "add" + to_string(max_idx) + "clear", clear_error_points(structure), colors);
	}

	save_nvm_file(data_name + "_multi", img_names, K, rotations, motions, structure, colors, key_points_for_all, correspond_struct_idx);
	remove_error_points(structure,correspond_struct_idx);
	cout << "after remove error points size :" << structure.size() << endl;
	save_ply_file(data_name + "multi", structure, colors);

	return 0;



	// 增量方式重建剩余的图像
	for (int i = 1; i < matches_for_all.size(); ++i)
	{
		//order_queryIdx 是已经重建过的照片，order_trainIdx是要添加进来的照片
		int order_queryIdx = img_add_order[i][0], order_trainIdx = img_add_order[i][1];
		cout << "add match " << order_queryIdx << " --- " << order_trainIdx << endl;
		vector<Point3d> object_points;
		vector<Point2f> image_points;
		Mat r, R, T;

		// 获取第order_queryIdx副图像中匹配点对应的三维点，以及在第order_trainIdx副图像中对应的像素点
		get_objpoints_and_imgpoints(
			matches_for_all[i],
			correspond_struct_idx[order_queryIdx],
			structure,
			key_points_for_all[order_trainIdx],
			object_points,
			image_points
		);

		// 求解变换矩阵
		bool pp = solvePnPRansac(object_points, image_points, K, noArray(), r, T);
		cout << "solvePnPRansac is " << pp << endl;
		// 将旋转向量转换为旋转矩阵
		Rodrigues(r, R);
		// 保存变换矩阵
		rotations[order_trainIdx] = R;
		motions[order_trainIdx] = T;
		//rotations.push_back(R);
		//motions.push_back(T);

		vector<Point2f> p1, p2;
		vector<Vec3b> c1, c2;
		get_matched_points(key_points_for_all[order_queryIdx], key_points_for_all[order_trainIdx], matches_for_all[i], p1, p2);
		get_matched_colors(colors_for_all[order_queryIdx], colors_for_all[order_trainIdx], matches_for_all[i], c1, c2);

		// 根据之前求得的R, T进行三维重建
		vector<Point3d> next_structure;
		reconstruct(K, rotations[order_queryIdx], motions[order_queryIdx], R, T, p1, p2, next_structure);

		//将新的重建结果与之前的融合
		fusion_structure(
			matches_for_all[i],
			correspond_struct_idx[order_queryIdx],
			correspond_struct_idx[order_trainIdx],
			structure,
			next_structure,
			colors,
			c1
		);

		//调试
		int tmp_max = 0;
		for (int i = 0; i < correspond_struct_idx.size();i++) {
			for (int j = 0; j < key_points_for_all[i].size(); j++) {
				if (correspond_struct_idx[i][j]>tmp_max) {
					tmp_max = correspond_struct_idx[i][j];
				}
			}
		}
		cout << "correspond struct idx max is :"<<tmp_max << endl;


		//这里进行增量重建的BA
		reconstructed_image_idx.push_back(order_trainIdx);
		Mat intrinsic_add(Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
		vector<Mat> extrinsics_add;
		for (size_t i = 0; i < reconstructed_image_idx.size(); ++i) {
			Mat extrinsic_add(6, 1, CV_64FC1);
			Mat r;
			Rodrigues(rotations[reconstructed_image_idx[i]], r);

			r.copyTo(extrinsic_add.rowRange(0, 3));
			motions[reconstructed_image_idx[i]].copyTo(extrinsic_add.rowRange(3, 6));

			extrinsics_add.push_back(extrinsic_add);
		}

		std::cout << "before add BA 3D size is:" << structure.size() << endl;

		bundle_adjustment(reconstructed_image_idx, intrinsic_add, extrinsics_add, correspond_struct_idx, key_points_for_all, structure);
		cout << intrinsic_add << endl;
		/*K.at<double>(0, 0) = intrinsic_add.at<double>(0, 0);
		K.at<double>(1, 1) = intrinsic_add.at<double>(1, 0);
		K.at<double>(0, 2) = intrinsic_add.at<double>(2, 0);
		K.at<double>(1, 2) = intrinsic_add.at<double>(3, 0);
		cout << K << endl;*/
		/*remove_error_points(structure);
		cout << "after remove error points 3D size is:" << structure.size() << endl;*/
		//save_ply_file("qhmadd" + to_string(order_trainIdx), structure, colors);

	}

	//save_camera_positions(rotations, motions);

	//Mat intrinsic(Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
	//vector<Mat> extrinsics;
	//for (size_t i = 0; i < rotations.size(); ++i) {
	//	Mat extrinsic(6, 1, CV_64FC1);
	//	Mat r;
	//	Rodrigues(rotations[i], r);

	//	r.copyTo(extrinsic.rowRange(0, 3));
	//	motions[i].copyTo(extrinsic.rowRange(3, 6));

	//	extrinsics.push_back(extrinsic);
	//}
	//cout << "before bundle adjustment" << endl;

	//bundle_adjustment(intrinsic, extrinsics, correspond_struct_idx, key_points_for_all, structure);
	save_nvm_file(data_name+"_multi", img_names, K, rotations, motions, structure, colors, key_points_for_all, correspond_struct_idx);

	remove_error_points(structure,correspond_struct_idx);
	cout << "after remove error points size :" << structure.size() << endl;

	//bundle_adjustment(intrinsic, extrinsics, correspond_struct_idx, key_points_for_all, structure);
	//打印tracks
	//cout << "track size is "<<correspond_struct_idx.size() << endl;
	//cout << "structure size is " << structure.size() << endl;
	//save_tracks(structure,key_points_for_all,correspond_struct_idx);

	//save_nvm_file("ET_multi", img_names, K, rotations, motions, structure, colors, key_points_for_all, correspond_struct_idx);
	//save_bundle_file("qhm_multi", K, rotations, motions, structure, colors, key_points_for_all, correspond_struct_idx);
	//保存
	save_ply_file(data_name+"multi", structure, colors);

	//save_nvm_file("qhm");
	save_structure("C:\\Users\\15297\\Desktop\\OpenCV_SFM\\Viewer\\structure.yml", rotations, motions, structure, colors);
	cout << "successful!!!" << endl;
	getchar();
	return 0;
}


void register_image(int order_queryIdx,int order_trainIdx) {

}
