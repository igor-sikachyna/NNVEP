#pragma once
#include <iostream>
#include <string>
#include <variant>
#include <functional>
#include <map>

namespace NNVEP
{
	class node
	{
	public:
		std::string name = "";
		int num_inputs = 1;
		using param = std::variant<int, double, std::string>;
		using params_map = std::map<std::string, param>;
		params_map params;
		std::function< std::map<std::string, param>() > get_default_params;
		std::function< void(params_map params) > run;
	};

	node getVideoFrameExtractionNode(std::string input, std::string output);
	node getSRNDeblurNode(std::string input, std::string output);
	node getESRGANNode(std::string input, std::string output);
	node getSRGANTensorflowNode(std::string input, std::string output);
	node getFrameResizeNode(std::string input, std::string output);
	node getDeblurGANNode(std::string input, std::string output);
	node getPytorchSepconvNode(std::string input, std::string output);
	node getReduceFPSNode(std::string input, std::string output, int old_fps, int new_fps);
	node getFolderCompareNode(std::string input1, std::string input2, std::string output);
	node getFrameRepairNode(std::string input1, std::string input2, std::string output, int sections_count, double max_allowed_change_coef);
	node getVideoCreatorNode(std::string input, std::string output_folder, std::string output_file_name, int output_fps);
	node getComparisonVideoCreatorNode(std::string input_file1, std::string input_file2, std::string output_folder, std::string output_file_name);
	node getCopyAudioToVideoNode(std::string input_video_without_audio, std::string input_video_with_audio, std::string output_folder, std::string output_file_name);

	class factory
	{
	public:
		void addNode(node n)
		{
			nodes.push_back(n);
		}

		node& operator[](size_t id)
		{
			return nodes[id];
		}

		node& last()
		{
			return nodes[nodes.size() - 1];
		}

		void run()
		{
			std::cout << "Starting process..." << std::endl;
			for (size_t i = 0; i < nodes.size(); i++)
			{
				std::cout << i + 1 << ") ";
				std::cout << nodes[i].name << ": ";
				if (nodes[i].num_inputs == 1) {
					std::cout << std::get<std::string>(nodes[i].params["input"]);
				}
				else
				{
					for (int j = 1; j <= nodes[i].num_inputs; j++)
					{
						std::cout << std::get<std::string>(nodes[i].params["input" + std::to_string(j)]);
						if (j != nodes[i].num_inputs) std::cout << " + ";
					}
				}
				std::cout << " -> " << std::get<std::string>(nodes[i].params["output"]) << std::endl;

				for (auto it = nodes[i].params.begin();
					it != nodes[i].params.end(); ++it)
				{
					std::cout << "\t"<< it->first << ": ";
					switch (it->second.index())
					{
					case 0:
						std::cout << std::get<int>(it->second);
						break;
					case 1:
						std::cout << std::get<double>(it->second);
						break;
					case 2:
						std::cout << std::get<std::string>(it->second);
						break;
					default:
						break;
					}
					std::cout << std::endl;
				}
			}

			std::cout << std::endl << "----------------------------------------" << std::endl << std::endl;

			std::cout << "<Press enter to continue>";
			std::cin.get();

			for (size_t i = 0; i < nodes.size(); i++)
			{
				std::cout << i + 1 << ") Started " << nodes[i].name << " node:" << std::endl;
				nodes[i].run(nodes[i].params);
			}
			std::cout << "Process finished!" << std::endl;
		}
	private:
		std::vector<node> nodes;
	};

	extern std::string python_command;
	extern std::string workspace_path;
}