#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>


using namespace cv;
using namespace std;


const string left_name = "f:/tmp/dataset/sequences/00/image_0/000000.png";
const string right_name = "f:/tmp/dataset/sequences/00/image_1/000000.png";


void refineDisparityRes(Mat & img_l, Mat & img_r, const Point2f & leftcorn, Point2f & rightcorn);

int main()
{
    ocl::setUseOpenCL(!cv::ocl::useOpenCL());
    
    VideoCapture cap_l(left_name);
    VideoCapture cap_r(right_name);

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);
    sgbm->setPreFilterCap(63);
    int sgbmWinSize = 3;
    sgbm->setBlockSize(sgbmWinSize);

    int cn = 1;

    sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(128);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    //sgbm->setMode(StereoSGBM::MODE_HH);
    sgbm->setMode(StereoSGBM::MODE_SGBM);
    //sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);
    Mat disp, disp32f;


    cv::Mat img_l, img_r, debug_img;
    while (cap_l.read(img_l) && cap_r.read(img_r))
    {
        std::vector<cv::Point2f> corners, right_corners, rc_disp;
        vector<uint8_t> status;
        vector<float> err;
        goodFeaturesToTrack(img_l, corners, 1000, 0.005, 10.0, noArray(), 5, false);
        calcOpticalFlowPyrLK(img_l.getUMat(ACCESS_READ), img_r.getUMat(ACCESS_READ), corners, right_corners, status, err, cv::Size(11, 11), 3);
        
        sgbm->compute(img_l, img_r, disp);
        disp.convertTo(disp32f, CV_32F, 1 / (16. * 255));

        cvtColor(img_l, debug_img, CV_GRAY2BGR);


        for (auto & f : corners)
        {
            circle(debug_img, f, 3, cv::Scalar(255, 0, 0), -1);
            float disp = disp32f.at<float>(int(f.y), int(f.x)) * 255.0f;
            rc_disp.push_back(Point2f(f.x - disp, f.y));
        }

        for (size_t i = 0; i < status.size(); ++i)
        {
            if (status[i] && abs(right_corners[i].y - corners[i].y) < 1.0f)
            {
                line(debug_img, right_corners[i], corners[i], cv::Scalar(0, 255, 0));
                //line(debug_img, rc_disp[i], corners[i], cv::Scalar(0, 255, 0));
                cv::circle(debug_img, right_corners[i], 3, cv::Scalar(0, 0, 255), -1);
                cv::circle(debug_img, rc_disp[i], 3, cv::Scalar(0, 255, 255), -1);
                refineDisparityRes(img_l, img_r, corners[i], rc_disp[i]);

            }
        }

        imshow("debug img", debug_img);
        imshow("disparity", disp32f);

        int key = waitKey(1);
        if (key == 27)
            break;

    }
    return 0;

}


float getColorSubpix(const cv::Mat& img, cv::Point2f pt)
{
    assert(!img.empty());

    int x = (int)pt.x;
    int y = (int)pt.y;

    int x0 = cv::borderInterpolate(x, img.cols, cv::BORDER_REFLECT_101);
    int x1 = cv::borderInterpolate(x + 1, img.cols, cv::BORDER_REFLECT_101);
    int y0 = cv::borderInterpolate(y, img.rows, cv::BORDER_REFLECT_101);
    int y1 = cv::borderInterpolate(y + 1, img.rows, cv::BORDER_REFLECT_101);

    float a = pt.x - (float)x;
    float c = pt.y - (float)y;

    float ret = (img.at<uint8_t>(y0, x0) * (1.f - a) + img.at<uint8_t>(y0, x1) * a) * (1.f - c)
        + (img.at<uint8_t>(y1, x0) * (1.f - a) + img.at<uint8_t>(y1, x1) * a) * c;

    return ret;
}

int64_t getCensus(const cv::Mat & img, const cv::Point2f & loc)
{
    int c_w = 9;
    int c_h = 5;
    int64_t census = 0;
    float center = getColorSubpix(img, loc);
    int num = 0;
    for (int j = -c_h / 2; j < c_h / 2 + 1; ++j)
    {
        float cy = loc.y + j;
        for (int i = -c_w / 2; i < c_w / 2 + 1; ++i)
        {
            if (i == 0 && j == 0) continue;
            float cx = loc.x + i;
            if (getColorSubpix(img, Point2f(cx, cy)) < center)
                census += int64_t(1) << num;
            ++num;
        }

    }
    return census;
}

float errCensus(int64_t c1, int64_t c2)
{
    return float(__popcnt64(c1^c2));
}

void refineDisparityRes(Mat & img_l, Mat & img_r, const Point2f & leftcorn, Point2f & rightcorn)
{
    int64_t census_left = getCensus(img_l, leftcorn);
    float h = 0.5f;
    Point2f iter_var = rightcorn;
    float err_start = errCensus(census_left, getCensus(img_r, iter_var));
    int iter_c = 5;
    //cout << "\n \n Original: " << err_start << "  " << rightcorn << endl;
    for (int k = 0; k < iter_c; ++k)
    {
        int64_t census_right_delta2 = getCensus(img_r, iter_var + Point2f(2 * h, 0.0f));
        int64_t census_right_delta = getCensus(img_r, iter_var + Point2f(h, 0.0f));
        int64_t census_right = getCensus(img_r, iter_var);
        float f_x_2h = errCensus(census_left, census_right_delta2);
        float f_x_h = errCensus(census_left, census_right_delta);
        float f_x = errCensus(census_left, census_right);

        if (f_x < err_start)
        {
            rightcorn = iter_var;
            err_start = f_x;
            //cout << k << ": " << f_x << " " << iter_var << endl;
        }

        float dx = (f_x_h - f_x) / h * h;
        float dxx = (f_x_2h - 2 * f_x_h + f_x) / (h * h);
        if (dxx == 0)
            break;
        float delta = -dx / dxx;
        iter_var.x += delta;
    }

}
