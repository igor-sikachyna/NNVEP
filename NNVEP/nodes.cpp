#include "nodes.h"
#include "helper.h"
#include <Windows.h>
#include <fstream>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

namespace NNVEP
{
	std::string python_command = "python";
	std::string workspace_path = "../workspace/";
	node getVideoFrameExtractionNode(std::string input, std::string output)
	{
		node res;
		res.name = "Frame extraction";
		res.get_default_params = [input, output]()
		{
			node::params_map params;
			params.insert(std::make_pair("input", workspace_path + input));
			params.insert(std::make_pair("output", workspace_path + output));
			params.insert(std::make_pair("frame_begin", 0));
			params.insert(std::make_pair("frame_end", -1));
			params.insert(std::make_pair("delete_output_contents", 1));
			params.insert(std::make_pair("resize_coef_x", 1.0));
			params.insert(std::make_pair("resize_coef_y", 1.0));
			params.insert(std::make_pair("skip_frames", 0));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::string input_file = std::get<std::string>(params["input"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			CreateDirectory(output_folder.c_str(), NULL);
			int frame_begin = std::get<int>(params["frame_begin"]);
			int frame_end = std::get<int>(params["frame_end"]);
			double scale_coef_x = std::get<double>(params["resize_coef_x"]);
			double scale_coef_y = std::get<double>(params["resize_coef_y"]);
			int skip_frames = std::get<int>(params["skip_frames"]);
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
			if (frame_end < 0)
			{
				frame_end = cap.get(cv::CAP_PROP_FRAME_COUNT);
			}
			int skip_counter = 0;
			while (!frame.empty())
			{
				if (frame_num >= frame_begin) {
					if (frame_num <= frame_end)
					{
						if (scale_coef_x != 1.0 || scale_coef_y != 1.0) {
							cv::Mat tframe;
							cv::resize(frame, tframe, cv::Size(), scale_coef_x, scale_coef_y);
							frame = tframe.clone();
						}

						if (skip_counter >= skip_frames)
						{
							skip_counter = 0;
							cv::imwrite(output_folder + "/" + std::to_string(save_frame_num) + ".png", frame);
							save_frame_num++;
							printProgressbar((save_frame_num * 100) / ((frame_end - frame_begin) / (skip_frames + 1)));
						}
						else
						{
							skip_counter++;
						}
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
		res.name = "SRN-Deblur";
		res.get_default_params = [input, output]()
		{
			node::params_map params;
			params.insert(std::make_pair("input", workspace_path + input));
			params.insert(std::make_pair("output", workspace_path + output));
			params.insert(std::make_pair("delete_output_contents", 1));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::string input_folder = std::get<std::string>(params["input"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			CreateDirectory(output_folder.c_str(), NULL);
			int delete_output_contents = std::get<int>(params["delete_output_contents"]);
			if (delete_output_contents == 1)
			{
				for (const auto & entry : std::filesystem::directory_iterator(output_folder))
				{
					std::filesystem::remove(entry.path());
				}
			}
			setLongNaming(input_folder);
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
		res.name = "ESRGAN";
		res.get_default_params = [input, output]()
		{
			node::params_map params;
			params.insert(std::make_pair("input", workspace_path + input));
			params.insert(std::make_pair("output", workspace_path + output));
			params.insert(std::make_pair("delete_output_contents", 1));
			params.insert(std::make_pair("model", "RRDB_PSNR_x4.pth"));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::string input_folder = std::get<std::string>(params["input"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			CreateDirectory(output_folder.c_str(), NULL);
			std::string model_name = std::get<std::string>(params["model"]);
			int delete_output_contents = std::get<int>(params["delete_output_contents"]);
			if (delete_output_contents == 1)
			{
				for (const auto & entry : std::filesystem::directory_iterator(output_folder))
				{
					std::filesystem::remove(entry.path());
				}
			}
			setLongNaming(input_folder);
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
		res.name = "SRGAN-Tensorflow";
		res.get_default_params = [input, output]()
		{
			node::params_map params;
			params.insert(std::make_pair("input", workspace_path + input));
			params.insert(std::make_pair("output", workspace_path + output));
			params.insert(std::make_pair("delete_output_contents", 1));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::string input_folder = std::get<std::string>(params["input"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			CreateDirectory(output_folder.c_str(), NULL);
			int delete_output_contents = std::get<int>(params["delete_output_contents"]);
			if (delete_output_contents == 1)
			{
				for (const auto & entry : std::filesystem::directory_iterator(output_folder))
				{
					std::filesystem::remove(entry.path());
				}
			}
			setLongNaming(input_folder);
			std::string cmd = "cd ../NNVEP-SRGAN-tensorflow & ";
			cmd += python_command + " main.py --output_dir " + output_folder + " --summary_dir ./result/log --mode inference --is_training False --task SRGAN --input_dir_LR " + input_folder + " --num_resblock 16 --perceptual_mode VGG54 --pre_trained_model True --checkpoint ./SRGAN_pre-trained/model-200000 & ";
			cmd += "cd ../NNVEP";
			system(cmd.c_str());
		};

		return res;
	}

	node getFrameResizeNode(std::string input, std::string output)
	{
		node res;
		res.name = "Frame resize";
		res.get_default_params = [input, output]()
		{
			node::params_map params;
			params.insert(std::make_pair("input", workspace_path + input));
			params.insert(std::make_pair("output", workspace_path + output));
			params.insert(std::make_pair("delete_output_contents", 1));
			params.insert(std::make_pair("resize_coef_x", 1.0));
			params.insert(std::make_pair("resize_coef_y", 1.0));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::string input_folder = std::get<std::string>(params["input"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			CreateDirectory(output_folder.c_str(), NULL);
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

			int total_files = 0;
			for (const auto & entry : std::filesystem::directory_iterator(input_folder))
			{
				total_files++;
			}

			int current_file = 1;
			for (const auto & entry : std::filesystem::directory_iterator(input_folder))
			{
				cv::Mat img;
				img = cv::imread(entry.path().string());
				cv::Mat resized;
				cv::resize(img, resized, cv::Size(), scale_coef_x, scale_coef_y);
				std::string new_file_name = output_folder + "/" + entry.path().filename().string();
				cv::imwrite(new_file_name, resized);
				printProgressbar((current_file * 100) / total_files);
				current_file++;
			}
		};

		return res;
	}

	node getDeblurGANNode(std::string input, std::string output)
	{
		node res;
		res.name = "DeblurGAN";
		res.get_default_params = [input, output]()
		{
			node::params_map params;
			params.insert(std::make_pair("input", workspace_path + input));
			params.insert(std::make_pair("output", workspace_path + output));
			params.insert(std::make_pair("delete_output_contents", 1));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::string input_folder = std::get<std::string>(params["input"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			CreateDirectory(output_folder.c_str(), NULL);
			int delete_output_contents = std::get<int>(params["delete_output_contents"]);
			if (delete_output_contents == 1)
			{
				for (const auto & entry : std::filesystem::directory_iterator(output_folder))
				{
					std::filesystem::remove(entry.path());
				}
			}
			setLongNaming(input_folder);
			int imgWidth = 0;
			int imgHeight = 0;
			for (const auto & entry : std::filesystem::directory_iterator(input_folder))
			{
				cv::Mat img;
				img = cv::imread(entry.path().string());
				imgWidth = img.cols;
				imgHeight = img.rows;
				break;
			}

			int num_files = 0;
			for (const auto & entry : std::filesystem::directory_iterator(input_folder))
			{
				num_files++;
			}

			std::string cmd = "cd ../NNVEP-DeblurGAN & ";
			cmd += "call %ANACONDA_DIR%/Scripts/activate deblurgan & ";
			cmd += python_command + " test.py --dataroot " + input_folder + " --results_dir " + output_folder + " --model test --dataset_mode single --learn_residual --display_id 0 --resize_or_crop resize --loadSizeX " + std::to_string(imgWidth) + " --loadSizeY " + std::to_string(imgHeight) + " --how_many " + std::to_string(num_files) + " & ";
			cmd += "call %ANACONDA_DIR%/Scripts/deactivate & ";
			cmd += "cd ../NNVEP";
			system(cmd.c_str());
		};

		return res;
	}

	node getPytorchSepconvNode(std::string input, std::string output)
	{
		node res;
		res.name = "pytorch-sepconv";
		res.get_default_params = [input, output]()
		{
			node::params_map params;
			params.insert(std::make_pair("input", workspace_path + input));
			params.insert(std::make_pair("output", workspace_path + output));
			params.insert(std::make_pair("delete_output_contents", 1));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::string input_folder = std::get<std::string>(params["input"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			CreateDirectory(output_folder.c_str(), NULL);
			int delete_output_contents = std::get<int>(params["delete_output_contents"]);
			if (delete_output_contents == 1)
			{
				for (const auto & entry : std::filesystem::directory_iterator(output_folder))
				{
					std::filesystem::remove(entry.path());
				}
			}
			setShortNaming(input_folder);
			std::string cmd = "cd ../NNVEP-pytorch-sepconv & ";
			cmd += python_command + " run.py --model l1 --input_directory " + input_folder + " --out " + output_folder + " & ";
			cmd += "cd ../NNVEP";
			system(cmd.c_str());
		};

		return res;
	}

	node getReduceFPSNode(std::string input, std::string output, int old_fps, int new_fps)
	{
		node res;
		res.name = "Reduce FPS";
		res.get_default_params = [input, output, old_fps, new_fps]()
		{
			node::params_map params;
			params.insert(std::make_pair("input", workspace_path + input));
			params.insert(std::make_pair("output", workspace_path + output));
			params.insert(std::make_pair("old_fps", old_fps));
			params.insert(std::make_pair("new_fps", new_fps));
			params.insert(std::make_pair("delete_output_contents", 1));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::string input_folder = std::get<std::string>(params["input"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			int old_fps = std::get<int>(params["old_fps"]);
			int new_fps = std::get<int>(params["new_fps"]);
			CreateDirectory(output_folder.c_str(), NULL);
			int delete_output_contents = std::get<int>(params["delete_output_contents"]);
			if (delete_output_contents == 1)
			{
				for (const auto & entry : std::filesystem::directory_iterator(output_folder))
				{
					std::filesystem::remove(entry.path());
				}
			}

			setLongNaming(input_folder);
			int total_files = 0;
			for (const auto & entry : std::filesystem::directory_iterator(input_folder))
			{
				total_files++;
			}
			int old_id = -1;
			int counter = 0;
			for (const auto & entry : std::filesystem::directory_iterator(input_folder))
			{
				int new_id = double(counter) * double(new_fps) / double(old_fps);
				if (new_id != old_id)
				{
					old_id = new_id;
					std::string file_name = entry.path().filename().string();
					std::filesystem::copy(entry.path(), output_folder + "/" + file_name);
				}
				counter++;
				printProgressbar((counter * 100) / total_files);
			}

		};

		return res;
	}

	node getFolderCompareNode(std::string input1, std::string input2, std::string output)
	{
		node res;
		res.name = "Folder compare";
		res.num_inputs = 2;
		res.get_default_params = [input1, input2, output]()
		{
			node::params_map params;
			params.insert(std::make_pair("input1", workspace_path + input1));
			params.insert(std::make_pair("input2", workspace_path + input2));
			params.insert(std::make_pair("output", workspace_path + output));
			params.insert(std::make_pair("delete_output_contents", 1));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::string input1_folder = std::get<std::string>(params["input1"]);
			std::string input2_folder = std::get<std::string>(params["input2"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			CreateDirectory(output_folder.c_str(), NULL);
			int delete_output_contents = std::get<int>(params["delete_output_contents"]);
			if (delete_output_contents == 1)
			{
				for (const auto & entry : std::filesystem::directory_iterator(output_folder))
				{
					std::filesystem::remove(entry.path());
				}
			}

			int total_files = 0;
			for (const auto & entry : std::filesystem::directory_iterator(input1_folder))
			{
				total_files++;
			}

			int current_file = 1;
			for (const auto & entry : std::filesystem::directory_iterator(input1_folder))
			{
				std::string file_name = entry.path().filename().string();
				if (std::filesystem::exists(input2_folder + "/" + file_name))
				{
					std::string name1 = entry.path().filename().stem().string();
					std::string name2 = output_folder + "/" + name1 + "_2" + entry.path().filename().extension().string();
					name1 += "_1" + entry.path().filename().extension().string();
					name1 = output_folder + "/" + name1;
					std::filesystem::copy(entry.path(), name1);
					std::filesystem::copy(input2_folder + "/" + file_name, name2);
				}
				printProgressbar((current_file * 100) / total_files);
				current_file++;
			}
		};

		return res;
	}

	node getFrameRepairNode(std::string input1, std::string input2, std::string output, int sections_count, double max_allowed_change_coef)
	{
		node res;
		res.name = "Frame repair";
		res.num_inputs = 2;
		res.get_default_params = [input1, input2, output, sections_count, max_allowed_change_coef]()
		{
			node::params_map params;
			params.insert(std::make_pair("input1", workspace_path + input1));
			params.insert(std::make_pair("input2", workspace_path + input2));
			params.insert(std::make_pair("output", workspace_path + output));
			params.insert(std::make_pair("delete_output_contents", 1));
			params.insert(std::make_pair("max_allowed_change_coef", max_allowed_change_coef));
			params.insert(std::make_pair("sections_count", sections_count));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::string input1_folder = std::get<std::string>(params["input1"]);
			std::string input2_folder = std::get<std::string>(params["input2"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			CreateDirectory(output_folder.c_str(), NULL);
			int delete_output_contents = std::get<int>(params["delete_output_contents"]);
			double max_allowed_change_coef = std::get<double>(params["max_allowed_change_coef"]);
			int sections_count = std::get<int>(params["sections_count"]);
			if (delete_output_contents == 1)
			{
				for (const auto & entry : std::filesystem::directory_iterator(output_folder))
				{
					std::filesystem::remove(entry.path());
				}
			}

			int total_files = 0;
			for (const auto & entry : std::filesystem::directory_iterator(input1_folder))
			{
				total_files++;
			}

			int current_file = 1;
			for (const auto & entry : std::filesystem::directory_iterator(input1_folder))
			{
				std::string file_name = entry.path().filename().string();
				if (std::filesystem::exists(input2_folder + "/" + file_name))
				{
					cv::Mat img_original = cv::imread(entry.path().string());
					cv::Mat img_changed = cv::imread(input2_folder + "/" + file_name);
					int step_size = max(max(img_changed.cols, img_changed.rows) / sections_count, 1);
					fix_image(img_original, img_changed, step_size, max_allowed_change_coef);
					std::string output_name = output_folder + "/" + entry.path().filename().stem().string() + entry.path().filename().extension().string();
					cv::imwrite(output_name, img_changed);
				}
				printProgressbar((current_file * 100) / total_files);
				current_file++;
			}
		};

		return res;
	}

	node getVideoCreatorNode(std::string input, std::string output_folder, std::string output_file_name, int output_fps)
	{
		node res;
		res.name = "Video creator";
		res.num_inputs = 1;
		res.get_default_params = [input, output_folder, output_file_name, output_fps]()
		{
			node::params_map params;
			params.insert(std::make_pair("input", workspace_path + input));
			params.insert(std::make_pair("output", workspace_path + output_folder));
			params.insert(std::make_pair("output_file", output_file_name));
			params.insert(std::make_pair("delete_output_contents", 0));
			params.insert(std::make_pair("output_fps", output_fps));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::string input_folder = std::get<std::string>(params["input"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			std::string output_file = std::get<std::string>(params["output_file"]);
			std::string full_output_file = output_folder + "/" + output_file;
			int delete_output_contents = std::get<int>(params["delete_output_contents"]);
			int output_fps = std::get<int>(params["output_fps"]);
			CreateDirectory(output_folder.c_str(), NULL);
			if (delete_output_contents == 1)
			{
				for (const auto & entry : std::filesystem::directory_iterator(output_folder))
				{
					std::filesystem::remove(entry.path());
				}
			}

			int frame_width = 0, frame_height = 0;
			for (const auto & entry : std::filesystem::directory_iterator(input_folder))
			{
				cv::Mat first_frame;
				first_frame = cv::imread(entry.path().string());
				frame_width = first_frame.cols;
				frame_height = first_frame.rows;
				break;
			}

			cv::VideoWriter video(full_output_file, CV_FOURCC('X', 'V', 'I', 'D'), output_fps, cv::Size(frame_width, frame_height), true);

			int total_files = 0;
			for (const auto & entry : std::filesystem::directory_iterator(input_folder))
			{
				total_files++;
			}

			int current_file = 1;
			for (const auto & entry : std::filesystem::directory_iterator(input_folder))
			{
				std::string file_name = entry.path().filename().string();
				cv::Mat frame = cv::imread(entry.path().string());
				if (!frame.empty())
				{
					video.write(frame);
				}
				printProgressbar((current_file * 100) / total_files);
				current_file++;
			}
		};

		return res;
	}

	node getComparisonVideoCreatorNode(std::string input_file1, std::string input_file2, std::string output_folder, std::string output_file_name)
	{
		node res;
		res.name = "Comparison video creator";
		res.num_inputs = 2;
		res.get_default_params = [input_file1, input_file2, output_folder, output_file_name]()
		{
			node::params_map params;
			params.insert(std::make_pair("input1", workspace_path + input_file1));
			params.insert(std::make_pair("input2", workspace_path + input_file2));
			params.insert(std::make_pair("output", workspace_path + output_folder));
			params.insert(std::make_pair("output_file", output_file_name));
			params.insert(std::make_pair("delete_output_contents", 0));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::string input1_file = std::get<std::string>(params["input1"]);
			std::string input2_file = std::get<std::string>(params["input2"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			std::string output_file = std::get<std::string>(params["output_file"]);
			std::string full_output_file = output_folder + "/" + output_file;
			int delete_output_contents = std::get<int>(params["delete_output_contents"]);
			CreateDirectory(output_folder.c_str(), NULL);
			if (delete_output_contents == 1)
			{
				for (const auto & entry : std::filesystem::directory_iterator(output_folder))
				{
					std::filesystem::remove(entry.path());
				}
			}

			cv::VideoCapture vid1(input1_file);
			cv::VideoCapture vid2(input2_file);

			int fps1 = vid1.get(CV_CAP_PROP_FPS);
			int fps2 = vid2.get(CV_CAP_PROP_FPS);

			int frame1_width = vid1.get(CV_CAP_PROP_FRAME_WIDTH), frame1_height = vid1.get(CV_CAP_PROP_FRAME_HEIGHT);
			int frame2_width = vid2.get(CV_CAP_PROP_FRAME_WIDTH), frame2_height = vid2.get(CV_CAP_PROP_FRAME_HEIGHT);

			int max_width = max(frame1_width, frame2_width);
			int max_height = max(frame1_height, frame2_height);

			if (fps2 > fps1)
			{
				std::swap(vid1, vid2);
				std::swap(fps1, fps2);
			}

			cv::VideoWriter video(full_output_file, CV_FOURCC('X', 'V', 'I', 'D'), fps1, cv::Size(max_width, max_height * 2), true);

			int total1_frames = vid1.get(CV_CAP_PROP_FRAME_COUNT);
			int total2_frames = vid2.get(CV_CAP_PROP_FRAME_COUNT);

			cv::Mat frame;
			cv::Mat old_frame;
			int frame_iterator = 0;
			for (int i = 0; i < total1_frames; i++)
			{
				vid1 >> frame;
				frame_iterator += fps1;
				if (frame_iterator >= fps2)
				{
					frame_iterator -= fps2;
					vid2 >> old_frame;
				}

				cv::Mat res_frame, res_old_frame;

				cv::Mat final_frame(max_height * 2, max_width, CV_8UC3, cv::Scalar(0, 0, 0));
				if (!frame.empty())
				{
					cv::resize(frame, res_frame, cv::Size(max_width, max_height));
					res_frame.copyTo(final_frame.rowRange(0, max_height).colRange(0, max_width));
				}
				if (!old_frame.empty())
				{
					cv::resize(old_frame, res_old_frame, cv::Size(max_width, max_height));
					res_old_frame.copyTo(final_frame.rowRange(max_height - 1, max_height * 2 - 1).colRange(0, max_width));
				}
				video.write(final_frame);
				printProgressbar((i * 100) / total1_frames);
			}
		};

		return res;
	}

	node getCopyAudioToVideoNode(std::string input_video_without_audio, std::string input_video_with_audio, std::string output_folder, std::string output_file_name)
	{
		node res;
		res.name = "Copy audio to video";
		res.num_inputs = 2;
		res.get_default_params = [input_video_without_audio, input_video_with_audio, output_folder, output_file_name]()
		{
			node::params_map params;
			params.insert(std::make_pair("input1", workspace_path + input_video_without_audio));
			params.insert(std::make_pair("input2", workspace_path + input_video_with_audio));
			params.insert(std::make_pair("output", workspace_path + output_folder));
			params.insert(std::make_pair("output_file", output_file_name));
			params.insert(std::make_pair("delete_output_contents", 0));
			return params;
		};

		res.params = res.get_default_params();
		res.run = [](node::params_map params)
		{
			std::string input_video_without_audio = std::get<std::string>(params["input1"]);
			std::string input_video_with_audio = std::get<std::string>(params["input2"]);
			std::string output_folder = std::get<std::string>(params["output"]);
			std::string output_file = std::get<std::string>(params["output_file"]);
			std::string full_output_file = output_folder + "/" + output_file;
			int delete_output_contents = std::get<int>(params["delete_output_contents"]);
			CreateDirectory(output_folder.c_str(), NULL);
			if (delete_output_contents == 1)
			{
				for (const auto & entry : std::filesystem::directory_iterator(output_folder))
				{
					std::filesystem::remove(entry.path());
				}
			}

			std::string cmd = "ffmpeg -i " + input_video_without_audio + " -i " + input_video_with_audio + " -map 0:v -map 1:a -c copy -shortest " + full_output_file;
			system(cmd.c_str());
		};

		return res;
	}
}