#ifndef DENSITY_H
#define DESNITY_H

#include "delaunator.hpp"
#include <cstdio>
#include <iostream>
#include <chrono>
#include <fstream>
#include <math.h>
#include <algorithm>

std::vector<std::pair<int,int>> getContiguousVoronoiCell(delaunator::Delaunator & d, std::vector<double> & coords, int point){
    double px = coords[2*point];
    double py = coords[2*point+1];
    std::vector<std::pair<int,int>> cell;
    for(std::size_t i = 0; i < d.triangles.size(); i+=3) {
        double tx0 = d.coords[2 * d.triangles[i]];
        double ty0 = d.coords[2 * d.triangles[i] + 1];
        double tx1 = d.coords[2 * d.triangles[i + 1]];
        double ty1 = d.coords[2 * d.triangles[i + 1] + 1];
        double tx2 = d.coords[2 * d.triangles[i + 2]];
        double ty2 = d.coords[2 * d.triangles[i + 2] + 1];

        if (px == tx0 && py == ty0){
            cell.push_back(std::pair<int,int>(i,0));
        }
        else if (px == tx1 && py == ty1){
            cell.push_back(std::pair<int,int>(i,1));
        }
        else if (px == tx2 && py == ty2){
            cell.push_back(std::pair<int,int>(i,2));
        }
    }
    return cell;
}

double lawOfCosine(double a, double b, double c){
    return std::acos(std::max(-1.0,std::min(1.0,(std::pow(a,2) + std::pow(b,2) - std::pow(c,2)) / (2.0*a*b+1e-100))));
}

std::vector<double> triangleAngles(double ax, double ay, double bx, double by, double cx, double cy){
    double a = std::sqrt( std::pow(cx-bx,2) + std::pow(cy-by,2) );
    double b = std::sqrt( std::pow(ax-cx,2) + std::pow(ay-cy,2) );
    double c = std::sqrt( std::pow(ax-bx,2) + std::pow(ay-by,2) );

    double A = lawOfCosine(b,c,a);
    double B = lawOfCosine(a,c,b);
    double C = M_PI - (A+B);

    return std::vector<double>({A,B,C});
}

double areaOfTriangle(double ax, double ay, double bx, double by, double cx, double cy){
    double a = std::sqrt( std::pow(cx-bx,2) + std::pow(cy-by,2) );
    double b = std::sqrt( std::pow(ax-cx,2) + std::pow(ay-cy,2) );
    double c = std::sqrt( std::pow(ax-bx,2) + std::pow(ay-by,2) );

    double s = (a+b+c)/2.0;

    s = s*(s-a)*(s-b)*(s-c);
    if (s < 0.0){
        return 0.0;
    }
    return std::sqrt(s);
}

double WeightedDTFELocalDensity(delaunator::Delaunator & d, std::vector<double> & coords, int point){
    std::vector<std::pair<int,int>> cell = getContiguousVoronoiCell(d,coords,point);

    double W = 0.0;
    double AW = 0.0;

    for (int i = 0; i < cell.size(); i++){
        int j = cell[i].first;
        double tx0 = d.coords[2 * d.triangles[j]];
        double ty0 = d.coords[2 * d.triangles[j] + 1];
        double tx1 = d.coords[2 * d.triangles[j + 1]];
        double ty1 = d.coords[2 * d.triangles[j + 1] + 1];
        double tx2 = d.coords[2 * d.triangles[j + 2]];
        double ty2 = d.coords[2 * d.triangles[j + 2] + 1];

        double a = areaOfTriangle(tx0,ty0,tx1,ty1,tx2,ty2);
        std::vector<double> angles = triangleAngles(tx0,ty0,tx1,ty1,tx2,ty2);
        double w = angles[cell[i].second];

        W += w;
        AW += (2.0*w*a);
    }
    return W / AW;
}

#endif
