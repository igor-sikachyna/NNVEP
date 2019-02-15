#include "nodes.h"
#include "helper.h"
#include <Windows.h>
#include <fstream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace NNVEP
{
	std::string python_command = "python";
	node getVideoFrameExtractionNode(std::string input, std::string output)
	{
		node res;
		res.get_default_params = [input, output]()
		{
			node::params_map params;
			params.insert(std::make_pair("input", input));
			params.insert(std::make_pair("output", output));
			params.insert(std::make_pair("frame_begin", 0));
			params.insert(std::make_pair("frame_end", -1));
			params.insert(std::make_pair("delete_output_contents", 1));
			params.insert(std::make_pair("resize_coef_x", 1.0));
			params.insert(std::make_pair("resize_coef_y", 1.0));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::cout << "Started frame extraction node: " << std::endl;
			std::string input_file = std::get<std::string>(params["input"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			int frame_begin = std::get<int>(params["frame_begin"]);
			int frame_end = std::get<int>(params["frame_end"]);
			double scale_coef_x = std::get<double>(params["resize_coef_x"]);
			double scale_coef_y = std::get<double>(params["resize_coef_y"]);
			int delete_output_contents = std::get<int>(params["delete_output_contents"]);
			if (delete_output_contents == 1)
			{
				for (const auto & entry : std::filesystem::directory_iterator(output_folder))
				{
					std::filesystem::remove(entry.path());
				}
			}
			cv::VideoCapture cap(input_file);
			cv::Mat frame;
			cap >> frame;
			int frame_num = 0;
			int save_frame_num = 0;
			while (!frame.empty())
			{
				if (frame_num >= frame_begin) {
					if (frame_end < 0 || frame_num <= frame_end)
					{
						if (scale_coef_x != 1.0 || scale_coef_y != 1.0) {
							cv::Mat tframe;
							cv::resize(frame, tframe, cv::Size(), scale_coef_x, scale_coef_y);
							frame = tframe.clone();
						}
						
						cv::imwrite(output_folder + "/" + std::to_string(save_frame_num) + ".png", frame);
						save_frame_num++;
						printProgressbar((save_frame_num * 100) / (frame_end - frame_begin));
					}
					else
					{
						break;
					}
				}
				frame_num++;
				cap >> frame;
			}
		};

		return res;
	}

	node getSRNDeblurNode(std::string input, std::string output)
	{
		node res;
		res.get_default_params = [input, output]()
		{
			node::params_map params;
			params.insert(std::make_pair("input", input));
			params.insert(std::make_pair("output", output));
			params.insert(std::make_pair("delete_output_contents", 1));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::cout << "Started SRN-Deblur node: " << std::endl;
			std::string input_folder = std::get<std::string>(params["input"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			int delete_output_contents = std::get<int>(params["delete_output_contents"]);
			if (delete_output_contents == 1)
			{
				for (const auto & entry : std::filesystem::directory_iterator(output_folder))
				{
					std::filesystem::remove(entry.path());
				}
			}

			std::string cmd = "cd ../NNVEP-SRN-Deblur & ";
			cmd += python_command + " run_model.py --input_path=" + input_folder + " --output_path=" + output_folder + " --gpu=0 & ";
			cmd += "cd ../NNVEP";
			system(cmd.c_str());
		};

		return res;
	}

	node getESRGANNode(std::string input, std::string output)
	{
		node res;
		res.get_default_params = [input, output]()
		{
			node::params_map params;
			params.insert(std::make_pair("input", input));
			params.insert(std::make_pair("output", output));
			params.insert(std::make_pair("delete_output_contents", 1));
			params.insert(std::make_pair("model", "RRDB_PSNR_x4.pth"));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::cout << "Started ESRGAN node: " << std::endl;
			std::string input_folder = std::get<std::string>(params["input"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			std::string model_name = std::get<std::string>(params["model"]);
			int delete_output_contents = std::get<int>(params["delete_output_contents"]);
			if (delete_output_contents == 1)
			{
				for (const auto & entry : std::filesystem::directory_iterator(output_folder))
				{
					std::filesystem::remove(entry.path());
				}
			}

			std::string cmd = "cd ../NNVEP-ESRGAN & ";
			cmd += python_command + " test.py models/" + model_name + " " + input_folder + " " + output_folder + " & ";
			cmd += "cd ../NNVEP";
			system(cmd.c_str());
		};

		return res;
	}

	node getSRGANTensorflowNode(std::string input, std::string output)
	{
		node res;
		res.get_default_params = [input, output]()
		{
			node::params_map params;
			params.insert(std::make_pair("input", input));
			params.insert(std::make_pair("output", output));
			params.insert(std::make_pair("delete_output_contents", 1));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::cout << "Started SRGAN-Tensorflow node: " << std::endl;
			std::string input_folder = std::get<std::string>(params["input"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			int delete_output_contents = std::get<int>(params["delete_output_contents"]);
			if (delete_output_contents == 1)
			{
				for (const auto & entry : std::filesystem::directory_iterator(output_folder))
				{
					std::filesystem::remove(entry.path());
				}
			}

			std::string cmd = "cd ../NNVEP-SRGAN-tensorflow & ";
			cmd += python_command + " main.py --output_dir " + output_folder + " --summary_dir ./result/log --mode inference --is_training False --task SRGAN --input_dir_LR " + input_folder + " --num_resblock 16 --perceptual_mode VGG54 --pre_trained_model True --checkpoint ./SRGAN_pre-trained/model-200000 & ";
			cmd += "cd ../NNVEP";
			system(cmd.c_str());
		};

		return res;
	}
}