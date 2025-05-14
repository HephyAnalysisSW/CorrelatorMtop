#include "helpers.h"

std::mutex coutMutex;

void showLoadingBar(int total, int finished) {
    const int barWidth = 50;
    float progress = static_cast<float>(finished) / total;
    int pos = static_cast<int>(barWidth * progress);

    std::lock_guard<std::mutex> lock(coutMutex);
    std::cout << "\r[";
    for (int j = 0; j < barWidth; ++j) {
        if (j < pos) std::cout << "=";
        else if (j == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %";
    std::cout.flush();
}
