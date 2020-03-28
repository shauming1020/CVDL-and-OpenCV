
// CVDL_HW1Dlg.cpp : ��@��
//

#include "stdafx.h"
#include "CVDL_HW1.h"
#include "CVDL_HW1Dlg.h"
#include "afxdialogex.h"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\opencv.hpp"
#include "cstdio"
#include "iostream"
#include <opencv/highgui.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace cv;
using namespace std;

// �� App About �ϥ� CAboutDlg ��ܤ��

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// ��ܤ�����
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV �䴩

// �{���X��@
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


// CCVDL_HW1Dlg ��ܤ��



CCVDL_HW1Dlg::CCVDL_HW1Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_CVDL_HW1_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CCVDL_HW1Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CCVDL_HW1Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CCVDL_HW1Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CCVDL_HW1Dlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CCVDL_HW1Dlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CCVDL_HW1Dlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CCVDL_HW1Dlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON7, &CCVDL_HW1Dlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON6, &CCVDL_HW1Dlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON8, &CCVDL_HW1Dlg::OnBnClickedButton8)
END_MESSAGE_MAP()


// CCVDL_HW1Dlg �T���B�z�`��

BOOL CCVDL_HW1Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// �N [����...] �\���[�J�t�Υ\���C

	// IDM_ABOUTBOX �����b�t�ΩR�O�d�򤧤��C
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

	// �]�w����ܤ�����ϥܡC�����ε{�����D�������O��ܤ���ɡA
	// �ج[�|�۰ʱq�Ʀ��@�~
	SetIcon(m_hIcon, TRUE);			// �]�w�j�ϥ�
	SetIcon(m_hIcon, FALSE);		// �]�w�p�ϥ�

	// TODO: �b���[�J�B�~����l�]�w
	AllocConsole();
	freopen("CONOUT$", "w", stdout);

	return TRUE;  // �Ǧ^ TRUE�A���D�z�ﱱ��]�w�J�I
}

void CCVDL_HW1Dlg::OnSysCommand(UINT nID, LPARAM lParam)
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

// �p�G�N�̤p�ƫ��s�[�J�z����ܤ���A�z�ݭn�U�C���{���X�A
// �H�Kø�s�ϥܡC���ϥΤ��/�˵��Ҧ��� MFC ���ε{���A
// �ج[�|�۰ʧ������@�~�C

void CCVDL_HW1Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // ø�s���˸m���e

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// �N�ϥܸm����Τ�ݯx��
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// �yø�ϥ�
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// ��ϥΪ̩즲�̤p�Ƶ����ɡA
// �t�ΩI�s�o�ӥ\����o�����ܡC
HCURSOR CCVDL_HW1Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

// ===== 1. Calibration ======
Mat cameraMatrix, distCoeffs, map1, map2;
vector<Mat> rvecs, tvecs;

Mat imgs[15];
Mat view_gray;

int img_cnt = 15; // �Ϲ��ƶq
Size image_size; // �Ϲ��ؤo
Size board_size = Size(8, 11); // �O�l�ؤo

void ReadChessboards() {
	string path = "./images/CameraCalibration/";
	string dtype = ".bmp";

	for (int i = 0; i < img_cnt; i++) {
		string s = to_string(i+1);
		cout << ">loading: " << path + s + dtype << endl;
		imgs[i] = imread(path+s+dtype, CV_LOAD_IMAGE_COLOR);
	}                                        
}

// 1.1 Find Corners
void CCVDL_HW1Dlg::OnBnClickedButton1()
{
	destroyAllWindows();

	vector<Point2f> srcCandidateCorners;; // �Ȧs�˴��쪺���I
	vector<Point3f> dstCandidateCorners; // �Ȧs�O�W���I�T���y��
	vector<vector<Point2f>> m_srcPoints; // �O�s�˴��쪺�Ҧ����I
	vector<vector<Point3f>> m_dstPoints;  // �O�s�O�W���I�T���y��

	/* Ū���Ϥ� */
	ReadChessboards();  

	/* ���o�Ϥ���T */
	image_size.width = imgs[0].cols;
	image_size.height = imgs[0].rows;
	cout << "image_size.width = " << image_size.width << endl;
	cout << "image_size.height = " << image_size.height << endl;
	
	/* ��l�ƴѽL�T���T�� */
	for (int i = 0; i < board_size.height; i++) {
		for (int j = 0; j < board_size.width; j++) {
			dstCandidateCorners.push_back(Point3f(i, j, 0.0f));
		}
	}

	
	bool found;
	for (int i = 0; i< img_cnt; i++) {
		/* �������I */
		found = findChessboardCorners(imgs[i], board_size, srcCandidateCorners);

		cvtColor(imgs[i], view_gray, CV_RGB2GRAY);

		/* �ȹ�����T�� */
		TermCriteria param(TermCriteria::MAX_ITER + TermCriteria::EPS, 30, 0.1);
		cornerSubPix(view_gray, srcCandidateCorners, Size(11, 11), Size(-1, -1), param);

		m_srcPoints.push_back(srcCandidateCorners); // �����ӹϪ����I��m
		m_dstPoints.push_back(dstCandidateCorners); // �����ӹϪ��ѽL�T���T��

		drawChessboardCorners(imgs[i], board_size, Mat(srcCandidateCorners), found);

		cout << ">deal with drawChessboardCorners to image[" << i+1 << "/" << img_cnt << "]" << endl;
	}

	/* Show Image */
	for (int i = 0; i < img_cnt; i++) {
		namedWindow("Display window", 0); // Create a window for display.
		imshow("Display window", imgs[i]);
		waitKey(500);
	}

	/* �B��Щw��� */
	cout << "Done Calibration ... " << endl;
	calibrateCamera(m_dstPoints, m_srcPoints, image_size, cameraMatrix, distCoeffs, rvecs, tvecs);

}

void CCVDL_HW1Dlg::OnBnClickedButton2()
{
	cout << "Instrinsic Matrix :" << endl;
	cout << Mat::Mat(cameraMatrix) << endl;
	cout << endl;
}


void CCVDL_HW1Dlg::OnBnClickedButton3()
{
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	Mat extrinsic_matrix = Mat(3, 4, CV_32FC1, Scalar::all(0));

	int index = ((CComboBox*)GetDlgItem(IDC_COMBO1))->GetCurSel();
	Rodrigues(rvecs[index], rotation_matrix); // R:3X3
	double val;

	int widthLimit = extrinsic_matrix.channels() * rotation_matrix.cols;

	for (int i = 0; i < extrinsic_matrix.rows; i++) {
		for (int j = 0; j < widthLimit; j++) {
			val = rotation_matrix.at<double>(i, j);
			extrinsic_matrix.at<float>(i, j) = (float)val;
		}
	}

	widthLimit = extrinsic_matrix.channels() * tvecs[index].cols;
	int width = extrinsic_matrix.channels() * extrinsic_matrix.cols;

	for (int i = 0; i < extrinsic_matrix.rows; i++) {
		for (int j = 0; j < widthLimit; j++) {
			val = tvecs[index].at<double>(i, j);
			extrinsic_matrix.at<float>(i, j + width - 1) = (float)val;
		}
	}
	cout << "Extrinsic Matrix :" << endl;
	cout << Mat::Mat(extrinsic_matrix) << endl;
	cout << endl;

}


void CCVDL_HW1Dlg::OnBnClickedButton4()
{
	cout << "Distortion Matrix :" << endl;
	cout << Mat::Mat(distCoeffs) << endl;
	cout << endl;
}

/* 2.1  Augmented Reality */
void drawAR(Mat &img, vector<Point2f> &img_pt) {
	// �e�u
	line(img, img_pt[0], img_pt[1], Scalar(0, 0, 255), 3, 8, 0);
	line(img, img_pt[1], img_pt[2], Scalar(0, 0, 255), 3, 8, 0);
	line(img, img_pt[2], img_pt[3], Scalar(0, 0, 255), 3, 8, 0);
	line(img, img_pt[3], img_pt[0], Scalar(0, 0, 255), 3, 8, 0);

	line(img, img_pt[0], img_pt[4], Scalar(0, 0, 255), 3, 8, 0);
	line(img, img_pt[1], img_pt[4], Scalar(0, 0, 255), 3, 8, 0);
	line(img, img_pt[2], img_pt[4], Scalar(0, 0, 255), 3, 8, 0);
	line(img, img_pt[3], img_pt[4], Scalar(0, 0, 255), 3, 8, 0);
}

void CCVDL_HW1Dlg::OnBnClickedButton5()
{
	destroyAllWindows();

	ReadChessboards();
	const int N_PIC = 5;

	vector<Point3f> obj_pt;
	vector<Point2f> imgs_pt[N_PIC];

	// Pyramid
	double x, y, z;
	x = -1; y = -1; z = 0;
	obj_pt.push_back(cv::Point3f(x, y, z)); // point0
	x = -1; y =  1; z = 0;
	obj_pt.push_back(cv::Point3f(x, y, z)); // point1
	x =  1; y =  1; z = 0;
	obj_pt.push_back(cv::Point3f(x, y, z)); // point2
	x =  1; y = -1; z = 0;
	obj_pt.push_back(cv::Point3f(x, y, z)); // point3
	x = 0; y = 0; z = -2;
	obj_pt.push_back(cv::Point3f(x, y, z)); // point4
	
	for (int i = 0; i < N_PIC; i++) {
		projectPoints(obj_pt, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imgs_pt[i]);
		drawAR(imgs[i], imgs_pt[i]);
	}

	for (int i = 0; i < 5; i++) {
		namedWindow("Display window", 0); // Create a window for display.
		imshow("Display window", imgs[i]);
		waitKey(500);
	}
}

/* 3.1 Transforms: Rotation, Scaling, Translation */
Mat src, dst;
Point2f srcPnt[4];
int counting = 0;

void Read_Image(string name) {
	string path = "./images/";
	string dtype = ".png";
	src = imread(path+name+dtype);
}

void CCVDL_HW1Dlg::OnBnClickedButton6()
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

void CCVDL_HW1Dlg::OnBnClickedButton7()
{
	destroyAllWindows();

	Read_Image("OriginalPerspective");
	namedWindow("Transform_1", WINDOW_NORMAL);
	imshow("Transform_1", src);

	setMouseCallback("Transform_1", on_mouse, NULL);
}

/* 4.1 Find Contour */
Mat src_gray;
vector<vector<Point> > contours;

void CCVDL_HW1Dlg::OnBnClickedButton8()
{
	destroyAllWindows();

	/* Read images*/
	Read_Image("Contour");
	cvtColor(src, src_gray, CV_BGR2GRAY); // Convert to gray
	dst = imread("./images/Contour.png");

	/* find Contours */
	findContours(src_gray, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	Scalar color(0, 0, 255);
	drawContours(dst, contours, -1, color, 2, LINE_8);

	namedWindow("Original");
	imshow("Original", src);

	namedWindow("FindContours");
	imshow("FindContours", dst);
}
