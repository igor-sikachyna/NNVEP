#include "helper.h"
#include <iostream>

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
		int done = percent / 5;
		if (helper::prev_done == done) return;
		helper::prev_done = done;
		int not_done = 20 - percent / 5;
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
}