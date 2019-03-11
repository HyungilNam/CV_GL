//Bilinear interpolation

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include<opencv2/calib3d.hpp>
#include<cmath>

using namespace cv;
using namespace std;

int Rindex = 0;
int done = 0;
cv::Mat pts(3, 4, CV_64F);
Point2f points[4];

int counter;


//vector<Point2f> clickedP(4);
vector<Point2d> clickedP;
vector<Point2d> settingP;


//vector<Point2f> selectedPts;

//사이즈 및 최대, 최소 좌표 값구하기 위한 변수 
vector<Point2d> sizeT(4);
vector<Point2d> sizeINV(4);

double Xmin, Xmax, Ymin, Ymax;

void SearchSize(Mat src, double* Homodata, double*Hdata);
Vec3b BIfuntion(Mat src,int Cx,int Cy, double p,double q, double r);
void setMatbyMouse(int event, int x, int y, int, void*);


int main(int argc, const char *argv[])
{
	//1-1.이미지 불러오기
	Mat src = imread("building2.jpg");
	cv::Mat pointedInputImg = cv::Mat(src.size(), src.type());

	//1-2.윈도우
	namedWindow("Output", 0);
	imshow("Output", src);
	//waitKey(0);
	

	setMouseCallback("Output", setMatbyMouse, NULL);


	while (clickedP.size() != 4)
	{
		waitKey(10);
	}

	for (size_t i = 0; i < 4; i++)
	{
		circle(src, clickedP[i], 3, Scalar(0, 0, 255), -1);
	}

	imshow("Output", src);

	setMouseCallback("Output", setMatbyMouse, NULL);

	while (settingP.size() != 4)
	{
		waitKey(10);
	}

	Mat dst = src;
	for (size_t i = 0; i < 4; i++)
	{
		circle(dst, settingP[i], 3, Scalar(0, 0, 255), -1);
	}

	imshow("Output", dst);


	//2-2. Homography, bilinear interpolation, warping
	Mat Homo = findHomography(clickedP, settingP);
	double *Homodata = (double*)Homo.data;


	Mat H = findHomography(clickedP, settingP);
	double *Htarget = (double*)H.data;
	Mat Hinv = H.inv();
	cout << Hinv << endl;
	double *Hdata = (double*)Hinv.data;


	Mat Hinvinv = Hinv.inv();
	cout << Hinvinv << endl;


	//출력 사이즈 찾기
	SearchSize(src, Homodata,Hdata);


	//결과 영상 크기
	Mat result = Mat(Size(Xmax-Xmin, Ymax-Ymin), src.type(), Scalar(0));

	//원본에서 좌표 색 찾아오기
	for (int i = 0; i < result.rows; i++) {
		for (int j = 0; j < result.cols; j++) {
			
			
			//  Mat pt = Mat(3, 1, CV_64FC1);
			double p = Hdata[0] * (j + Xmin) + Hdata[1] * (i + Ymin) + Hdata[2];
			double q = Hdata[3] * (j + Xmin) + Hdata[4] * (i + Ymin) + Hdata[5];
			double r = Hdata[6] * (j + Xmin) + Hdata[7] * (i + Ymin) + Hdata[8];
			//  pt = H * pt;
			int Cx = floor(p / r);
			int Cy = floor(q / r);


			result.at<Vec3b>(i, j) = BIfuntion(src, Cx, Cy,p,q,r);
			
		}
		cv::imshow("Modified", result);
		waitKey(10);

	}
	waitKey(0);

	Mat result2;
	Mat H2 = findHomography(clickedP, settingP);
	warpPerspective(src, result2, H2, Size(Xmax - Xmin, Ymax - Ymin));

	cv::circle(result, cv::Point(pts.at<double>(0,0),pts.at<double>(1,0)), 2, cv::Scalar(0, 0, 255), -1);

	imshow("standard", result2);

	waitKey(0);

	return 0;
}


void SearchSize(Mat src, double* Homodata, double*Hdata)
{
	vector<Point2f> sizeP(4);
	sizeP[0].x = 0; sizeP[0].y = 0;
	sizeP[1].x = src.cols; sizeP[1].y = 0;
	sizeP[2].x = src.cols; sizeP[2].y = src.rows;
	sizeP[3].x = 0; sizeP[3].y = src.rows;


	for (int i = 0; i < 4; i++)
	{
		pts.at<double>(0, i) = sizeP[i].x;
		pts.at<double>(1, i) = sizeP[i].y;
		pts.at<double>(2, i) = 1.0;
	}
	
	Mat homoMat = cv::Mat(3, 3, CV_64F, Homodata);
	Mat homoMatInv = cv::Mat(3, 3, CV_64F, Hdata);


	pts = homoMat * pts;

	
	for (int i = 0; i < 4; i++) {	
		pts.at<double>(0, i) /= pts.at<double>(2, i);
		pts.at<double>(1, i) /= pts.at<double>(2, i);

		sizeINV[i].x = Hdata[0] * sizeT[i].y + Hdata[1] * sizeT[i].x + Hdata[2];
		sizeINV[i].y = Hdata[3] * sizeT[i].y + Hdata[4] * sizeT[i].x + Hdata[5];
		double zINV = Hdata[6] * sizeT[i].y + Hdata[7] * sizeT[i].x + Hdata[8];

		sizeINV[i].x = (sizeINV[i].x) / zINV;
		sizeINV[i].y = (sizeINV[i].y) / zINV;

		cout << sizeINV << endl;

	}
	//출력 사이즈 찾기-->min/max좌표 구하기
	Mat home0;// = cv::Mat(4, 1, CV_64F);
	Mat home1;// = cv::Mat(4, 1, CV_64F);
	home0 = pts.row(0);
	home1 = pts.row(1);
	//void minMaxLoc(InputArray src, double* minVal, double* maxVal=0, Point* minLoc=0, Point* maxLoc=0, InputArray mask=noArray())¶
	
	cv::minMaxLoc(home0, &Xmin, &Xmax);
	cv::minMaxLoc(home1, &Ymin, &Ymax);

}

Vec3b BIfuntion(Mat src, int Cx, int Cy, double p, double q, double r)
{
	if (Cx>0 && Cy >0 && Cx<src.cols - 1 && Cy<src.rows - 1) {
		//이중 선형 보간법
		double fx1 = p / r - (double)Cx;
		double fx2 = 1 - fx1;
		double fy1 = q / r - (double)Cy;
		double fy2 = 1 - fy1;

		double w1 = fx2*fy2;
		double w2 = fx1*fy2;
		double w3 = fx1*fy1;
		double w4 = fx2*fy1;

		Vec3b P1 = src.at<Vec3b>(Cy, Cx);
		Vec3b P2 = src.at<Vec3b>(Cy, Cx + 1);
		Vec3b P3 = src.at<Vec3b>(Cy + 1, Cx + 1);
		Vec3b P4 = src.at<Vec3b>(Cy + 1, Cx);

		return P1*w1 + P2*w2 + P3*w3 + P4*w4;
	}
	//원본 영역에 없는 거라면
	else {
		//검정색으로 채워주기
		return cv::Vec3b(0, 0, 0);
	}
}


void setMatbyMouse(int event, int x, int y, int, void*)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		counter++;
		if (clickedP.size() != 4)
		{
			clickedP.push_back(Point2f(x, y));
		}
		else
		{
			settingP.push_back(Point2f(x, y));
		}

		return;
	}
}