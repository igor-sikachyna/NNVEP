#pragma once
#include <iostream>
#include <opencv2/core.hpp>

namespace NNVEP
{
	void printProgressbar(int percent);
	void setLongNaming(std::string folder);
	void setShortNaming(std::string folder);
	bool isNumericalNaming(std::string folder);
	bool isLongNaming(std::string folder);
	bool isShortNaming(std::string folder);
	void fix_image(cv::Mat& original, cv::Mat& new_image, int lookup_area_size, double max_allowed_change);
}