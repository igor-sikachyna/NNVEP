#include <iostream>
#include "nodes.h"

int main()
{
	NNVEP::python_command = "python";

	NNVEP::factory process;
	process.addNode(NNVEP::getVideoFrameExtractionNode("../workspace/input/input.avi", "../workspace/input_frames"));
	process[0].params["frame_begin"] = 1848;
	process[0].params["frame_end"] = std::get<int>(process[0].params["frame_begin"]) + 10;
	//process[0].params["resize_coef_x"] = 0.5;
	//process[0].params["resize_coef_y"] = 0.5;

	process.addNode(NNVEP::getSRNDeblurNode("../workspace/input_frames", "../workspace/srn_deblur"));

	//process.addNode(NNVEP::getESRGANNode("../workspace/srn_deblur", "../workspace/esrgan"));
	process.addNode(NNVEP::getSRGANTensorflowNode("../workspace/srn_deblur", "../workspace/srgan-tf"));

	process.run();
	std::cout << "Process finished!" << std::endl;
	std::cin.get();
	return 0;
}