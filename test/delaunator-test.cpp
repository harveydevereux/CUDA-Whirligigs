#include "../include/density.h"

int main(int argc, char ** argv) {

    int N = 0;

    std::ifstream inputFile;
    inputFile.open(argv[1]);
    std::vector<double> coords;
    if (inputFile.is_open()){
        std::string line;
        while (std::getline(inputFile,line)){
            // x,y
            double x = 0.0;
            double y = 0.0;
            int c = 0;
            std::string lineStr = line;
            while (c < 2){
                if (c == 1){
                    y = std::stod(lineStr);
                    break;
                }
                size_t position = lineStr.find(",");
                std::string str = lineStr.substr(0,position);
                if (c == 0){
                    x = std::stod(str);
                }
                lineStr = lineStr.substr(position+1);
                c += 1;
            }
            coords.push_back(x);
            coords.push_back(y);
            N += 1;
        }
    }
    else{
        std::cout << "Could not open input file: " << argv[1] << std::endl;
        return -1;
    }


    /* x0, y0, x1, y1, ... */

    auto t1 = std::chrono::high_resolution_clock::now();

    //triangulation happens here
    delaunator::Delaunator d(coords);

    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    std::cout << "Triangulation complete in: " << duration * 1e-6 << " seconds\n";

    std::ofstream out("out.txt");

    std::vector<double> cells(N);
    for (int i = 0; i < N; i++){
        cells[i] = WeightedDTFELocalDensity(d,coords,i);
        out << cells[i] << std::endl;
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>( t3 - t1 ).count();

    std::cout << "density complete in: " << duration2 * 1e-6 << " seconds\n";
}
