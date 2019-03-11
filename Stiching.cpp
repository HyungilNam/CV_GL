//stitching

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include<opencv2/calib3d.hpp>
#include<cmath>
using namespace cv;
using namespace std;

//사이즈 및 최대, 최소 좌표 값구하기 위한 변수 
vector<Point2f> sizeT(4);
vector<Point2f> sizeINV(4);
double Xmin, Xmax, Ymin, Ymax;
cv::Mat pts(3, 4, CV_64F);
void SearchSize(Mat LeftImg, Mat RightImg, double* Homodata, double*Hdata);

Vec3b BIfuntion(Mat src, int Cx, int Cy, double p, double q, double r);

double ForHomo[] = { 1.962, -0.172, -635.756, 0.557, 1.738, -255.159, 0.00159, 0, 1 };
Mat Homography(3, 3, CV_64FC1, ForHomo);

double FindOverlap(Mat pts, Mat RightImg);

int main()
{
	Mat LeftImg = imread("img2.jpg");
	Mat RightImg = imread("img3.jpg");

	imshow("Output1",LeftImg);
	imshow("Output2", RightImg);

	//출력 사이즈 찾기
	double *Homodata = (double*)Homography.data;

	Mat Hinv = Homography.inv();
	double *Hdata = (double*)Hinv.data;

	SearchSize(LeftImg,RightImg,Homodata, Hdata);

	//결과 영상 크기
	Mat resultLeft = Mat(Size(Xmax - Xmin, Ymax - Ymin), LeftImg.type(), Scalar(0));

	//원본에서 좌표 색 찾아오기
	for (int i = 0; i < resultLeft.rows; i++) {
		for (int j = 0 ; j < resultLeft.cols; j++) {
			
			//  Mat pt = Mat(3, 1, CV_64FC1);
			double p = Hdata[0] * (j + Xmin) + Hdata[1] * (i + Ymin) + Hdata[2];
			double q = Hdata[3] * (j + Xmin) + Hdata[4] * (i + Ymin) + Hdata[5];
			double r = Hdata[6] * (j + Xmin) + Hdata[7] * (i + Ymin) + Hdata[8];
			//  pt = H * pt;
			int Cx = floor(p / r);
			int Cy = floor(q / r);

			resultLeft.at<Vec3b>(i,j) = BIfuntion(LeftImg, Cx, Cy, p, q, r);
		}
		
	}
	cv::imshow("resultLeft", resultLeft);
//	cv::waitKey(0);

	//LeftImage를 Homography로 돌렸으니,이제 RightImg와 Blending
	//출력 결과물을 담을 변수 하나 만들기(크기 중요)

	//1.size를 다시 조정
	//각 이미지를 붙이기 위해서 빈 이미지 만들어서 위에다가 덮기 -->y축 맞추기
	int overlapwidth = (int)FindOverlap(pts, RightImg);

	int width = resultLeft.cols + RightImg.cols - overlapwidth;
	int height = resultLeft.rows;


	//resultLeft가 호모그라피된 이미지 --> 사이즈에 맞게 조정한게 resultLeft2
	Mat resultLeft2 = Mat(Size(width, height), LeftImg.type(), Scalar(0)); //homography뒤의 왼쪽 그림
	Mat resultRight = Mat(Size(width, height), LeftImg.type(), Scalar(0)); 

	//왼쪽 사진 붙이고 
	Mat imageROIL(resultLeft2, Rect(0, 0, resultLeft.cols, resultLeft.rows));
	resultLeft.copyTo(imageROIL);


	//1-2을 위한 사전 조치
	Mat imageROIRR(resultRight, Rect(width - RightImg.cols, -(Ymin), RightImg.cols, RightImg.rows));
	RightImg.copyTo(imageROIRR);

	//공통영역 만들기!!
	Mat LeftWhite = Mat(Size(width, height), LeftImg.type(), Scalar(0)); //일단 검은 배경
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (resultLeft2.at<Vec3b>(y, x) != Vec3b(0, 0, 0) && resultRight.at<Vec3b>(y, x) != Vec3b(0, 0, 0))
			{
				LeftWhite.at<Vec3b>(y, x) = Vec3b(225, 225, 225);
			}
		}
	}

	imshow("middle", LeftWhite);
	waitKey(0);
	
	//1-2. blending하기
	//Feathering 실습자료 기반한 부분
	Mat mask1 = Mat::zeros(height,width,CV_32F); //32F하면 0-1로만 구성 가능!
	Mat mask2 = Mat::zeros(height,width,CV_32F);
	
	//Rect(x좌표,y좌표, x길이,y길이)
	rectangle(mask1, Rect(0, 0,  resultLeft.cols - overlapwidth, height), Scalar(1.0f), FILLED);
	rectangle(mask2, Rect(RightImg.cols + overlapwidth, 0, RightImg.cols, height), Scalar(1.0f), FILLED);

	for (int y = 0; y < height; y++)
	{
		int widthcounter = 0;
		for (int counterIndex = resultLeft.cols - overlapwidth; counterIndex < resultLeft.cols; counterIndex++)
		{
			if (LeftWhite.at<Vec3b>(y, counterIndex) == Vec3b(225, 225, 225))
			{
				widthcounter++;
			}
		}
		for (int ox = 0; ox < overlapwidth; ox++)
		{
			int x = resultLeft.cols - overlapwidth + ox;
			if (LeftWhite.at<Vec3b>(y, x) == Vec3b(225, 225, 225))
			{
				mask2.at<float>(y, x) = 1.0f*ox / widthcounter;
				mask1.at<float>(y, x) = 1.0f - mask2.at<float>(y, x);
			}
		}
	}
/*
	for (int ox = 0; ox < overlapwidth; ox++)
	{
		int x = resultLeft.cols - overlapwidth + ox;
		for (int y = 0; y < height; y++)
		{
			int widthcounter = 0;
			if (LeftWhite.at<Vec3b>(y,x) == Vec3b(225,225,225))
			{
				widthcounter++;
				mask2.at<float>(y, x) = 1.0f*(y)/overlapwidth * ox;
				mask1.at<float>(y, x) = 1.0f - mask2.at<float>(y, x);
			}
		}
	}
*/
	Mat resultFinal = Mat(height, width, CV_8UC3);
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (LeftWhite.at<Vec3b>(y, x) == Vec3b(225, 225, 225))
			{
				resultFinal.at<Vec3b>(y, x) = resultLeft2.at<Vec3b>(y, x)*mask1.at<float>(y, x) +
					resultRight.at<Vec3b>(y, x)*mask2.at<float>(y, x);
			}
			else
			{
				if (resultLeft2.at<Vec3b>(y, x) == Vec3b(0, 0, 0))
				{
					resultFinal.at<Vec3b>(y, x) = resultRight.at<Vec3b>(y, x);
				}
				else
				{
					resultFinal.at<Vec3b>(y, x) = resultLeft2.at<Vec3b>(y, x);
				}
			}
		}
	}
	/*
	double wL = 0.8;
	double wR = 1 - wL;
	for (int i = 0; i < height; i++)
	{
		for (int j = width - RightImg.cols; j < width; j++)
		{
			if (j>=resultLeft.cols)
			{
				resultFinal.at<Vec3b>(i, j) = resultRight.at<Vec3b>(i, j);
			}
			else {
				if (resultFinal.at<Vec3b>(i, j) == Vec3b(0, 0, 0) && resultRight.at<Vec3b>(i,j) != Vec3b(0,0,0))
				{
					resultFinal.at<Vec3b>(i, j) = resultRight.at<Vec3b>(i, j);
				}
				else if(resultFinal.at<Vec3b>(i, j) != Vec3b(0, 0, 0) && resultRight.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
				{
					resultFinal.at<Vec3b>(i, j) = wL * resultFinal.at<Vec3b>(i, j) + wR * resultRight.at<Vec3b>(i, j);
				}
				
			}
		}
	*/
	imshow("색 혼합!", resultFinal);
	waitKey(0);

	return 0;
}

void SearchSize(Mat LeftImg, Mat RightImg, double* Homodata, double*Hdata)
{
	vector<Point2f> sizeL(4);
	vector<Point2f> sizeR(4);

	sizeL[0].x = 0; sizeL[0].y = 0;
	sizeL[1].x = LeftImg.cols; sizeL[1].y = 0;
	sizeL[2].x = LeftImg.cols; sizeL[2].y = LeftImg.rows;
	sizeL[3].x = 0; sizeL[3].y = LeftImg.rows;

	sizeR[0].x = 0; sizeR[0].y = 0;
	sizeR[1].x = RightImg.cols; sizeR[1].y = 0;
	sizeR[2].x = RightImg.cols; sizeR[2].y = RightImg.rows;
	sizeR[3].x = 0; sizeR[3].y = RightImg.rows;

	for (int i = 0; i < 4; i++)
	{
		pts.at<double>(0, i) = sizeL[i].x;
		pts.at<double>(1, i) = sizeL[i].y;
		pts.at<double>(2, i) = 1.0;
	}

	Mat homoMat = cv::Mat(3, 3, CV_64F, Homodata);
	Mat homoMatInv = cv::Mat(3, 3, CV_64F, Hdata);

	pts = homoMat * pts;
	
	for (int i = 0; i < 4; i++) {
		pts.at<double>(0, i) /= pts.at<double>(2, i);
		pts.at<double>(1, i) /= pts.at<double>(2, i);

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

double FindOverlap(Mat pts, Mat RightImg)
{
	return pts.at<double>(0, 1);
}
