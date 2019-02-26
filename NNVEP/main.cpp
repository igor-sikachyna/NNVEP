#include <iostream>
#include "nodes.h"
#include "helper.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cmath>

int main()
{
	NNVEP::python_command = "python";
	NNVEP::workspace_path = "../workspace/";

	NNVEP::factory process;
	process.addNode(NNVEP::getVideoFrameExtractionNode("../workspace/input/IntroX.avi", "../workspace/input_frames"));
	//process[0].params["frame_begin"] = 936;
	//process[0].params["frame_end"] = std::get<int>(process[0].params["frame_begin"]) + 10;
	
	//process.addNode(NNVEP::getVideoFrameExtractionNode("../workspace/input/OutroX.avi", "../workspace/input_frames"));
	//process[0].params["skip_frames"] = 23;
	//process[0].params["frame_begin"] = 1849;
	//process[0].params["frame_begin"] = 3360;
	//process[0].params["frame_end"] = std::get<int>(process[0].params["frame_begin"]) + 10;
	//process[0].params["resize_coef_x"] = 0.5;
	//process[0].params["resize_coef_y"] = 0.5;
	
	//process.addNode(NNVEP::getVideoFrameExtractionNode("../workspace/input/Friends.S01E01.The.One.Where.Monica.Gets.a.Roommate.mkv", "../workspace/input_frames"));
	//process[0].params["skip_frames"] = 230;

	process.addNode(NNVEP::getSRNDeblurNode("../workspace/input_frames", "../workspace/srn_deblur"));
	process.addNode(NNVEP::getDeblurGANNode("../workspace/input_frames", "../workspace/deblurgan"));
	process.addNode(NNVEP::getFrameRepairNode("../workspace/input_frames", "../workspace/srn_deblur", "../workspace/srn_deblur_repaired", 100, 0.07));
	process.addNode(NNVEP::getFrameRepairNode("../workspace/input_frames", "../workspace/deblurgan", "../workspace/deblurgan_repaired", 100, 0.07));
	process.addNode(NNVEP::getFrameRepairNode("../workspace/srn_deblur_repaired", "../workspace/deblurgan_repaired", "../workspace/deblurred", 100, 0.07));
	//process.addNode(NNVEP::getFolderCompareNode("../workspace/input_frames", "../workspace/deblurred", "../workspace/compare"));
	//process.addNode(NNVEP::getFolderCompareNode("../workspace/input_frames", "../workspace/deblurred_test", "../workspace/compare_test"));
	//process.addNode(NNVEP::getFrameResizeNode("deblurred", "deblurred_low"));
	//process.last().params["resize_coef_x"] = 0.9;
	//process.last().params["resize_coef_y"] = 0.9;
	//process.addNode(NNVEP::getESRGANNode("deblurred", "final"));

	//process.addNode(NNVEP::getVideoCreatorNode("final", "res_videos", "video1.avi", 24));
	//process.addNode(NNVEP::getComparisonVideoCreatorNode("res_videos/video1a.avi", "HumanOp/video1a.avi", "res_comparison", "video_comparison.mkv"));
	//process.addNode(NNVEP::getCopyAudioToVideoNode("res_videos/video1.avi", "input/HumanOp.avi", "res_videos", "video1a.avi"));
	//process.addNode(NNVEP::getCopyAudioToVideoNode("res_videos/video2.avi", "input/HumanOp.avi", "res_videos", "video2a.avi"));

	//process.addNode(NNVEP::getFolderCompareNode("../workspace/input_frames", "../workspace/deblurred", "../workspace/compare"));

	//process.addNode(NNVEP::getESRGANNode("../workspace/srn_deblur", "../workspace/esrgan"));
	//process.addNode(NNVEP::getSRGANTensorflowNode("../workspace/srn_deblur", "../workspace/srgan-tf"));

	//process.addNode(NNVEP::getFrameResizeNode("../workspace/esrgan", "../workspace/temp_resize"));
	//process.last().params["resize_coef_x"] = 0.25;
	//process.last().params["resize_coef_y"] = 0.25;

	//process.addNode(NNVEP::getESRGANNode("../workspace/temp_resize", "../workspace/esrgan_run_2"));

	process.run();
	std::cin.get();
	return 0;
}

int main_()
{
	cv::Mat img_original;
	cv::Mat img_deblurred1, img_deblurred2;
	img_original = cv::imread("../workspace/input_frames/139.png");
	img_deblurred1 = cv::imread("../workspace/compare/139_1.png");
	img_deblurred2 = cv::imread("../workspace/compare/139_2.png");
	cv::Mat img_fixed1 = img_deblurred1.clone();
	cv::Mat img_fixed2 = img_deblurred2.clone();
	int step = std::max(img_original.cols / 100, 1);
	NNVEP::fix_image(img_original, img_fixed1, step, 15.0);
	NNVEP::fix_image(img_original, img_fixed2, step, 10.0);
	cv::Mat try_final_fix = img_fixed2.clone();
	NNVEP::fix_image(img_fixed1, try_final_fix, step, 5.0);

	cv::namedWindow("display", CV_WINDOW_FREERATIO);

	int key = 0;
	int show_mode = 1;
	while (key != 27)
	{
		if (key >= 48 && key <= 57)
		{
			show_mode = key - 48;
		}

		if (show_mode == 1) cv::imshow("display", img_original);
		else if (show_mode == 2) cv::imshow("display", img_deblurred1);
		else if (show_mode == 3) cv::imshow("display", img_deblurred2);
		else if (show_mode == 4) cv::imshow("display", img_fixed1);
		else if (show_mode == 5) cv::imshow("display", img_fixed2);
		else if (show_mode == 6) cv::imshow("display", try_final_fix);
		key = cv::waitKey(1);
		if (key >= 0) std::cout << key << std::endl;
	}

	return 0;
}