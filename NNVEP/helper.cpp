#include "helper.h"
#include <iostream>
#include <filesystem>

namespace NNVEP
{
	namespace helper
	{
		int prev_percent = -1;
		int prev_done = -1;
	}

	void printProgressbar(int percent)
	{
		if (percent == helper::prev_percent || percent > 100) return;
		//if (percent > 100) percent = 100;
		int done = percent / 2;
		//if (helper::prev_done == done) return;
		helper::prev_done = done;
		int not_done = 50 - percent / 2;
		std::cout << "|";
		for (int i = 0; i < done; i++)
		{
			std::cout << "#";
		}
		for (int i = 0; i < not_done; i++)
		{
			std::cout << " ";
		}
		std::cout << "| " << percent << "%" << std::endl;
		helper::prev_percent = percent;
	}

	void setLongNaming(std::string folder)
	{
		if (!isNumericalNaming(folder)) return;
		if (isLongNaming(folder)) return;
		int longest_name = 0;
		for (const auto & entry : std::filesystem::directory_iterator(folder))
		{
			int new_file_length = entry.path().filename().string().length();
			if (new_file_length > longest_name) longest_name = new_file_length;
		}

		for (const auto & entry : std::filesystem::directory_iterator(folder))
		{
			int new_file_length = entry.path().filename().string().length();
			if (new_file_length != longest_name)
			{
				std::string new_name = "";
				for (int i = 0; i < longest_name - new_file_length; i++)
				{
					new_name += "0";
				}
				new_name += entry.path().filename().string();
				new_name = entry.path().parent_path().string() + "/" + new_name;
				std::filesystem::rename(entry.path(), new_name);
			}
		}
	}

	void setShortNaming(std::string folder)
	{
		if (!isNumericalNaming(folder)) return;
		if (isShortNaming(folder)) return;
		for (const auto & entry : std::filesystem::directory_iterator(folder))
		{
			std::string new_filename = entry.path().filename().string();
			if (entry.path().filename().stem().string() != "0")
			{
				int non_zero = 0;
				for (int i = 0; i < new_filename.size(); i++)
				{
					if (new_filename[i] != '0')
					{
						non_zero = i;
						break;
					}
				}

				if (non_zero > 0)
				{
					std::string name = "";
					for (int i = non_zero; i < new_filename.size(); i++)
					{
						name += new_filename[i];
					}
					name = entry.path().parent_path().string() + "/" + name;
					std::filesystem::rename(entry.path(), name);
				}
			}
		}
	}

	bool isNumericalNaming(std::string folder)
	{
		for (const auto & entry : std::filesystem::directory_iterator(folder))
		{
			std::string file_name = entry.path().filename().stem().string();
			for (int i = 0; i < file_name.length(); i++)
			{
				if (file_name[i] < '0' || file_name[i] > '9') return false;
			}
		}
		return true;
	}

	bool isLongNaming(std::string folder)
	{
		int prev_file_name_length = -1;
		for (const auto & entry : std::filesystem::directory_iterator(folder))
		{
			int new_file_length = entry.path().filename().string().length();
			if (prev_file_name_length < 0)
			{
				prev_file_name_length = new_file_length;
			}
			else
			{
				if (prev_file_name_length != new_file_length) return false;
			}
		}
		return true;
	}

	bool isShortNaming(std::string folder)
	{
		return !isLongNaming(folder);
	}

	void fix_image(cv::Mat& original, cv::Mat& new_image, int lookup_area_size, double max_allowed_change)
	{
		int step = std::max(lookup_area_size / 2, 1);
		for (int i = 0; i < new_image.cols - lookup_area_size; i += step)
		{
			for (int j = 0; j < new_image.rows - lookup_area_size; j += step)
			{
				double actual_change = 0.0;
				double original_avarage = 0.0;
				for (int ti = 0; ti < lookup_area_size; ti++)
				{
					for (int tj = 0; tj < lookup_area_size; tj++)
					{
						cv::Vec3b v_fixed = new_image.at<cv::Vec3b>(cv::Point(i + ti, j + tj));
						cv::Vec3b v_original = original.at<cv::Vec3b>(cv::Point(i + ti, j + tj));
						actual_change += sqrt(pow(int(v_fixed[0]) - int(v_original[0]), 2) + pow(int(v_fixed[1]) - int(v_original[1]), 2) + pow(int(v_fixed[2]) - int(v_original[2]), 2));
						original_avarage += (double(v_original[0]) + double(v_original[1]) + double(v_original[2])) / 3.0;
					}
				}
				actual_change /= double(lookup_area_size * lookup_area_size);
				original_avarage /= double(lookup_area_size * lookup_area_size);
				double coef = 1.0;
				if (actual_change > 0.0) coef = max_allowed_change * original_avarage / actual_change;
				if (coef > 1.0) coef = 1.0;
				//std::cout << poisson_coef << std::endl;
				for (int ti = 0; ti < lookup_area_size; ti++)
				{
					for (int tj = 0; tj < lookup_area_size; tj++)
					{
						cv::Vec3b v_fixed = new_image.at<cv::Vec3b>(cv::Point(i + ti, j + tj));
						cv::Vec3b v_original = original.at<cv::Vec3b>(cv::Point(i + ti, j + tj));
						for (int c = 0; c < 3; c++)
						{
							new_image.at<cv::Vec3b>(cv::Point(i + ti, j + tj))[c] = int(double(v_fixed[c]) * coef + double(v_original[c]) * (1.0 - coef));
						}
					}
				}
			}
		}
	}
}