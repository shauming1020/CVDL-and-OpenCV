
// OpenCV_HW1Dlg.cpp : 實作檔
//

#include "stdafx.h"
#include "OpenCV_HW1.h"
#include "OpenCV_HW1Dlg.h"
#include "afxdialogex.h"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\opencv.hpp"
#include "cstdio"
#include "iostream"
#include <opencv/highgui.h>
#include <math.h>


#ifdef _DEBUG
#define new DEBUG_NEW
#define M_PI 3.14159265358979323846  /* pi */
#endif

using namespace cv;
using namespace std;

// 對 App About 使用 CAboutDlg 對話方塊

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 對話方塊資料
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支援

// 程式碼實作
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// COpenCV_HW1Dlg 對話方塊



COpenCV_HW1Dlg::COpenCV_HW1Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_OPENCV_HW1_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void COpenCV_HW1Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(COpenCV_HW1Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &COpenCV_HW1Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &COpenCV_HW1Dlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &COpenCV_HW1Dlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &COpenCV_HW1Dlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &COpenCV_HW1Dlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &COpenCV_HW1Dlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON7, &COpenCV_HW1Dlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &COpenCV_HW1Dlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON9, &COpenCV_HW1Dlg::OnBnClickedButton9)
	ON_BN_CLICKED(IDC_BUTTON10, &COpenCV_HW1Dlg::OnBnClickedButton10)
	ON_BN_CLICKED(IDC_BUTTON11, &COpenCV_HW1Dlg::OnBnClickedButton11)
	ON_BN_CLICKED(IDC_BUTTON13, &COpenCV_HW1Dlg::OnBnClickedButton13)
END_MESSAGE_MAP()


// COpenCV_HW1Dlg 訊息處理常式

BOOL COpenCV_HW1Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 將 [關於...] 功能表加入系統功能表。

	// IDM_ABOUTBOX 必須在系統命令範圍之中。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 設定此對話方塊的圖示。當應用程式的主視窗不是對話方塊時，
	// 框架會自動從事此作業
	SetIcon(m_hIcon, TRUE);			// 設定大圖示
	SetIcon(m_hIcon, FALSE);		// 設定小圖示

	// TODO: 在此加入額外的初始設定
	AllocConsole();
	freopen("CONOUT$", "w", stdout);

	return TRUE;  // 傳回 TRUE，除非您對控制項設定焦點
}

void COpenCV_HW1Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果將最小化按鈕加入您的對話方塊，您需要下列的程式碼，
// 以便繪製圖示。對於使用文件/檢視模式的 MFC 應用程式，
// 框架會自動完成此作業。

void COpenCV_HW1Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 繪製的裝置內容

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 將圖示置中於用戶端矩形
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 描繪圖示
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 當使用者拖曳最小化視窗時，
// 系統呼叫這個功能取得游標顯示。
HCURSOR COpenCV_HW1Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

/* 1. Image Processing*/
string path = "./images/";
IplImage *img, *result_img;
Size img_size;

char *strToChar(string str) {
	char *ch = new char[str.length()];
	int len = str.length();
	strcpy(ch, str.c_str());
	return ch;
}

void Window_show(string window_name, IplImage *image) {
	char *show_name = strToChar(window_name);
	namedWindow(show_name, WINDOW_NORMAL); // Create a window for display.
	cvShowImage(show_name, image);
	cvWaitKey(1);
	cvReleaseImage(&image);
}

void ReadImage(string data_name, bool toGray=FALSE, int HW=1) {
	cout << ">loading: " << path + data_name << ", toGray: " << toGray << endl;
	char *readpath = strToChar(path + data_name);

	if (toGray == TRUE)
		img = cvLoadImage(readpath, CV_LOAD_IMAGE_GRAYSCALE);
	else
		img = cvLoadImage(readpath, CV_LOAD_IMAGE_COLOR);
	img_size.width = img->width;
	img_size.height = img->height;
	
	if (HW == 1) 
		result_img = cvCreateImage(img_size, IPL_DEPTH_8U, 3); // For hw1.
	else if (HW == 2)
		result_img = cvCreateImage(img_size, img->depth, img->nChannels); // For hw2.
}

/* 1.1 Load Image File  */

void COpenCV_HW1Dlg::OnBnClickedButton1()
{
	destroyAllWindows();
	ReadImage("dog.bmp");

	cout << "image_size.width = " << img_size.width << endl;
	cout << "image_size.height = " << img_size.height << endl;

	Window_show("OpenCV_hw1.1", img);
}

/* 1.2 Color Conversion */
void COpenCV_HW1Dlg::OnBnClickedButton2()
{
	destroyAllWindows();
	ReadImage("color.png");

	IplImage *image2_B = cvCreateImage(img_size, IPL_DEPTH_8U, 3);
	IplImage *image2_G = cvCreateImage(img_size, IPL_DEPTH_8U, 3);
	IplImage *image2_R = cvCreateImage(img_size, IPL_DEPTH_8U, 3);

	uchar color_B, color_G, color_R;

	for (int i = 0; i<img->height; i++) {
		for (int j = 0; j<img->widthStep; j = j + 3) {
			color_B = img->imageData[i*img->widthStep + j];
			color_G = img->imageData[i*img->widthStep + j + 1];
			color_R = img->imageData[i*img->widthStep + j + 2];

			image2_B->imageData[i*image2_B->widthStep + j] = color_B;
			image2_G->imageData[i*image2_G->widthStep + j + 1] = color_G;
			image2_R->imageData[i*image2_R->widthStep + j + 2] = color_R;

			result_img->imageData[i*result_img->widthStep + j] = color_G;
			result_img->imageData[i*result_img->widthStep + j + 1] = color_R;
			result_img->imageData[i*result_img->widthStep + j + 2] = color_B;
		}
	}
	Window_show("OpenCV_hw1.2-original", img);
	Window_show("OpenCV_hw1.2-result", result_img);

	cvReleaseImage(&image2_B);
	cvReleaseImage(&image2_G);
	cvReleaseImage(&image2_R);
}

/* 1.3 Image Flipping */

IplImage Flipping() {
	IplImage *image_B = cvCreateImage(img_size, IPL_DEPTH_8U, 3);
	IplImage *image_G = cvCreateImage(img_size, IPL_DEPTH_8U, 3);
	IplImage *image_R = cvCreateImage(img_size, IPL_DEPTH_8U, 3);

	IplImage *flipping_img = cvCreateImage(img_size, IPL_DEPTH_8U, 3);
	
	uchar color_B, color_G, color_R;

	for (int i = 0; i<img->height; i++) {
		for (int j = 0; j<img->widthStep; j = j + 3) {
			color_B = img->imageData[i*img->widthStep + j];
			color_G = img->imageData[i*img->widthStep + j + 1];
			color_R = img->imageData[i*img->widthStep + j + 2];

			image_B->imageData[i*image_B->widthStep + j] = color_B;
			image_G->imageData[i*image_G->widthStep + j + 1] = color_G;
			image_R->imageData[i*image_R->widthStep + j + 2] = color_R;

			flipping_img->imageData[i*flipping_img->widthStep + (flipping_img->widthStep - j - 3)] = color_B;
			flipping_img->imageData[i*flipping_img->widthStep + (flipping_img->widthStep - j - 2)] = color_G;
			flipping_img->imageData[i*flipping_img->widthStep + (flipping_img->widthStep - j - 1)] = color_R;
		}
	}
	cvReleaseImage(&image_B);
	cvReleaseImage(&image_G);
	cvReleaseImage(&image_R);
	return *flipping_img;
}

void COpenCV_HW1Dlg::OnBnClickedButton3()
{
	destroyAllWindows();
	ReadImage("dog.bmp");

	*result_img = Flipping();

	Window_show("OpenCV_hw1.3-original", img);
	Window_show("OpenCV_hw1.3-result", result_img);
}

/* 1.4 Blending */
const int alpha_slider_max = 100;
int alpha_slider;
int slider = 40;
int angle_slider = 10;
int flag = 0;
double alpha;
double beta;
IplImage *flp_img;

void my_change(int, void*)
{
	alpha = (double)alpha_slider / alpha_slider_max;
	beta = (1.0 - alpha);
	cvAddWeighted(img, alpha, flp_img, beta, 0.0, result_img);
	cvShowImage("Blend", result_img);
}


void COpenCV_HW1Dlg::OnBnClickedButton4()
{
	destroyAllWindows();

	ReadImage("dog.bmp");
	cvNamedWindow("Blend");
	alpha_slider = 0;
	flp_img = cvCreateImage(img_size, IPL_DEPTH_8U, 3);
	*flp_img = Flipping();

	char TrackbarName[50];
	sprintf(TrackbarName, "Alpha %d", alpha_slider_max);
	cvCreateTrackbar2(TrackbarName, "Blend", &alpha_slider, alpha_slider_max, my_change);

	my_change(alpha_slider, 0);

	cvWaitKey(0);
	
}

/* 2. Adaptive Threshold */
/* 2.1 Global Threshold */
void COpenCV_HW1Dlg::OnBnClickedButton5()
{
	destroyAllWindows();
	ReadImage("QR.png", TRUE, 2);
	
	cvThreshold(img, result_img, 80, 255, CV_THRESH_BINARY);

	Window_show("Original", img);
	Window_show("Threshold", result_img);

	cvWaitKey(0);
	cvReleaseImage(&img);
	cvReleaseImage(&result_img);
}

/* 2.2 Local Threshold */
void COpenCV_HW1Dlg::OnBnClickedButton6()
{
	destroyAllWindows();
	ReadImage("QR.png", TRUE, 2);

	cvAdaptiveThreshold(img, result_img, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 19, -1);

	Window_show("Original", img);
	Window_show("AdaptiveThreshold", result_img);

	cvWaitKey(0);
	cvReleaseImage(&img);
	cvReleaseImage(&result_img);

}

/* 4. Convolution */
Mat myFilter, afterConvImage;
Mat sobel_x, sobel_y, soble_mag;
float G_x, G_y, G_mag;

Mat creat_Gaussian_kernel(int height, int width, double sigma) {
	Mat kernel = Mat(height, width, CV_32FC1); // create channel matrices

	double sum = 0.0; // sum is for normalization 
	double r, s = 2.0 * sigma * sigma;
	int x, y;

	// generating height x width kernel 
	for (x = 0; x < height; x++) {
		for (y = 0; y < width; y++) {
			r = sqrt(x*x + y*y);
			kernel.at<float>(x, y ) = (float)((exp(-(r * r) / s)) / (M_PI * s));
			sum += kernel.at<float>(x, y);
		}
	}

	// normalising the Kernel 
	for (x = 0; x < height; x++) {
		for (y = 0; y < width; y++) {
			kernel.at<float>(x, y) /= sum;
		}
	}
	cout << "my kernel " << endl;
	cout << kernel << endl;

	return kernel;
}

Mat applyFilter(Mat image, Mat filter) {

	int height = image.rows;
	int width = image.cols;
	int filterHeight = filter.rows;
	int filterWidth = filter.cols;

	int newImageWidth = width - filterWidth + 1;
	int newImageHeight = height - filterHeight + 1;

	int i, j, h, w;

	Mat convImage = Mat::zeros(newImageHeight, newImageWidth, CV_32FC1);

	// Gaussian blur
	for (i = 0; i < newImageHeight; i++) {
		for (j = 0; j < newImageWidth; j++) {
			for (h = i; h < i + filterHeight; h++) {
				for (w = j; w < j + filterWidth; w++) {
					convImage.at<float>(i, j) += filter.at<float>(h - i, w - j) * image.at<float>(h, w);
				}
			}
		}
	}

	return convImage;
}

void applySobel(Mat convImage) {
	int convimg_height = convImage.rows;
	int convimg_width = convImage.cols;
	sobel_x = convImage.clone();
	sobel_y = convImage.clone();
	soble_mag = convImage.clone();

	for (int i = 1; i < convimg_height - 1; i++) {
		for (int j = 1; j <convimg_width - 1; j++) {
			G_x = abs(convImage.at<float>(i - 1, j + 1) + 2 * convImage.at<float>(i, j + 1) + convImage.at<float>(i + 1, j + 1)
				- (convImage.at<float>(i - 1, j - 1) + 2 * convImage.at<float>(i, j - 1) + convImage.at<float>(i + 1, j - 1)));
			G_y = abs(convImage.at<float>(i - 1, j - 1) + 2 * convImage.at<float>(i - 1, j) + convImage.at<float>(i - 1, j + 1)
				- (convImage.at<float>(i + 1, j - 1) + 2 * convImage.at<float>(i + 1, j) + convImage.at<float>(i + 1, j + 1)));
			G_mag = sqrt(G_x*G_x + G_y*G_y);

			sobel_x.at<float>(i, j) = (float)G_x;
			sobel_y.at<float>(i, j) = (float)G_y;
			soble_mag.at<float>(i, j) = (float)G_mag;
		}
	}
}

/* 4.1 Gaussian */
void COpenCV_HW1Dlg::OnBnClickedButton7()
{
	destroyAllWindows();

	Mat school;
	Size school_size;
	school = imread("./images/school.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	school_size.height = school.rows;
	school_size.width = school.cols;
	school.convertTo(school, CV_32FC1, 1.0 / 255.0);
	imshow("Original", school);

	myFilter = creat_Gaussian_kernel(3, 3, 1.0); // create Gaussian filter with 3x3, sigma = 1.0
	afterConvImage = applyFilter(school, myFilter);

	imshow("Blur", afterConvImage);

	applySobel(afterConvImage);
}


void COpenCV_HW1Dlg::OnBnClickedButton8()
{
	imshow("Sobel_X", sobel_x);
}


void COpenCV_HW1Dlg::OnBnClickedButton9()
{
	imshow("Sobel_Y", sobel_y);
}


void COpenCV_HW1Dlg::OnBnClickedButton10()
{
	imshow("Magitude", soble_mag);
}

/* 3.1 Transforms: Rotation, Scaling, Translation */
Mat src, dst;
Point2f srcPnt[4];
int counting = 0;

void Read_Image(string name) {
	string path = "./images/";
	string dtype = ".png";
	src = imread(path + name + dtype);
}

void COpenCV_HW1Dlg::OnBnClickedButton11()
{
	destroyAllWindows();

	Read_Image("OriginalTransform");

	Mat dst = Mat::zeros(src.rows, src.cols, src.type());
	Mat dst1 = Mat::zeros(src.rows, src.cols, src.type());
	CString ang, sca;
	int x, y;

	GetDlgItemText(IDC_EDIT1, ang.GetBuffer(10), 10);
	GetDlgItemText(IDC_EDIT2, sca.GetBuffer(10), 10);
	x = GetDlgItemInt(IDC_EDIT3);
	y = GetDlgItemInt(IDC_EDIT4);

	double angle = _wtof(ang);
	double scale = _wtof(sca);
	//printf("angle = %lf, scale = %lf, x = %d, y = %d\n", angle, scale, x, y);

	Point2f srcTri[3];
	srcTri[0] = Point2f(0, 0);
	srcTri[1] = Point2f(src.cols - 1, 0);
	srcTri[2] = Point2f(0, src.rows - 1);

	Point2f dstTri[3];
	dstTri[0] = Point2f(0 + x, 0 + y);
	dstTri[1] = Point2f(src.cols - 1 + x, 0 + y);
	dstTri[2] = Point2f(0 + x, src.rows - 1 + y);

	Mat warp_mat = getAffineTransform(srcTri, dstTri);
	warpAffine(src, dst, warp_mat, dst.size());

	Point center = Point(130 + x, 125 + y);
	Mat rot_mat = getRotationMatrix2D(center, angle, scale);
	warpAffine(src, dst, rot_mat, dst.size());

	/* Window Show */
	namedWindow("Origin", WINDOW_NORMAL);
	imshow("Origin", src);

	namedWindow("Transform", WINDOW_NORMAL);
	imshow("Transform", dst);
	waitKey(0);
}

/* 3.2 Perspective Transformation */
void on_mouse(int event, int x, int y, int flags, void* param) {

	if (event == CV_EVENT_LBUTTONDOWN) {
		srcPnt[counting] = Point2f(x, y);
		//printf("%d, %d\n", x, y);
		counting++;
		if (counting >= 4) {
			Point2f dstTri[4];
			dstTri[0] = Point2f(20, 20);
			dstTri[1] = Point2f(450, 20);
			dstTri[2] = Point2f(450, 450);
			dstTri[3] = Point2f(20, 450);

			Size size(450, 450);
			Mat perspective = getPerspectiveTransform(srcPnt, dstTri);
			warpPerspective(src, dst, perspective, size, INTER_LINEAR);
			namedWindow("Transform_2", WINDOW_NORMAL);
			imshow("Transform_2", dst);
			counting = 0;
		}
	}
}

void COpenCV_HW1Dlg::OnBnClickedButton13()
{
	destroyAllWindows();

	Read_Image("OriginalPerspective");
	namedWindow("Transform_1", WINDOW_NORMAL);
	imshow("Transform_1", src);

	setMouseCallback("Transform_1", on_mouse, NULL);
	cvWaitKey(0);

}
