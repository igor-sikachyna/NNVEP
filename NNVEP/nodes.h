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

		void run()
		{
			for (size_t i = 0; i < nodes.size(); i++)
			{
				nodes[i].run(nodes[i].params);
			}
		}
	private:
		std::vector<node> nodes;
	};

	extern std::string python_command;
}