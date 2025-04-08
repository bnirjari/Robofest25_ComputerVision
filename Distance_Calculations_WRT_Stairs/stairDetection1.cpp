#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{

    Mat img = imread("stair.png");
    imshow("Source",img);

    //Apply Gaussian blur to get good results
    GaussianBlur(img,img,Size(5,5),0,0);

    Mat dst, out_img,control;
    Canny(img, dst, 80, 240, 3);
    cvtColor(dst, out_img, CV_GRAY2BGR);
    cvtColor(dst, control, CV_GRAY2BGR);

    vector<int> y_keeper_for_lines;
    vector<Vec4i> lines;
    HoughLinesP(dst, lines, 1, CV_PI/180, 30, 40, 5 );

    for( size_t i = 1; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        line( control, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
    }

    Vec4i l = lines[0];
    line( out_img, Point(0, l[1]), Point(img.cols, l[1]), Scalar(0,0,255), 3, CV_AA);
    y_keeper_for_lines.push_back(l[1]);

    int okey = 1;
    int stair_counter = 1;

    for( size_t i = 1; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        for(int m:y_keeper_for_lines)
        {
            if(abs(m-l[1])<15)
                okey = 0;

        }
        if(okey)
        {
            line( out_img, Point(0, l[1]), Point(img.cols, l[1]), Scalar(0,0,255), 3, CV_AA);
            y_keeper_for_lines.push_back(l[1]);
            stair_counter++;
        }
        okey = 1;

    }
    putText(out_img,"Stair number:" + to_string(stair_counter),Point(40,60),FONT_HERSHEY_SIMPLEX,1.5,Scalar(0,255,0),2);
    imshow("Before", img);
    imshow("Control", control);
    imshow("detected lines", out_img);
    waitKey(0);
    return 0;
}