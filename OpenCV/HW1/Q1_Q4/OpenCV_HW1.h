
// OpenCV_HW1.h : PROJECT_NAME ���ε{�����D�n���Y��
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�� PCH �]�t���ɮ׫e���]�t 'stdafx.h'"
#endif

#include "resource.h"		// �D�n�Ÿ�


// COpenCV_HW1App: 
// �аѾ\��@�����O�� OpenCV_HW1.cpp
//

class COpenCV_HW1App : public CWinApp
{
public:
	COpenCV_HW1App();

// �мg
public:
	virtual BOOL InitInstance();

// �{���X��@

	DECLARE_MESSAGE_MAP()
};

extern COpenCV_HW1App theApp;