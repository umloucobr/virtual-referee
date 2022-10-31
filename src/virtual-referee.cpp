#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
	cv::namedWindow("Line Detector", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Debug", cv::WINDOW_AUTOSIZE);

	cv::VideoCapture video;
	video.open("t.mp4");

	cv::Mat vid1;
	cv::Mat vid2;
	cv::Mat vid2Gray;
	cv::Mat vid2Color;
	cv::Mat vid2Canny;
	cv::Ptr<cv::BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2(500, 16.0, false);
	cv::Mat pMOG2Mask;
	cv::Mat finalResult;
	std::vector<cv::Vec2f> lines;

	for (;;)
	{
		video >> vid1;
		if (vid1.empty())
		{
			return 0;
		}
		cv::GaussianBlur(vid1, vid2, cv::Size(5, 5), 3, 3);
		cv::cvtColor(vid2, vid2Gray, cv::COLOR_BGR2GRAY);
		cv::Canny(vid2Gray, vid2Canny, 50, 200, 3, true);

		cv::cvtColor(vid2Canny, vid2Color, cv::COLOR_GRAY2BGR);		
		cv::HoughLines(vid2Canny, lines, 1, CV_PI / 180, 150, 0, 0);

		for (size_t i = 0; i < lines.size(); i++)
		{
			float rho = lines[i][0];
			float theta = lines[i][1];

			cv::Point pt1;
			cv::Point pt2;

			double a = cos(theta), b = sin(theta);
			double x0 = a * rho, y0 = b * rho;

			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));
			line(vid2Color, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
		}

		pMOG2->apply(vid2, pMOG2Mask);

		cv::imshow("Line Detector", pMOG2Mask);
		cv::imshow("Debug", vid2Color);
		
		if (static_cast<int>(cv::waitKey(33)) == 27)
		{
			return 0;
		}
	}

	return 0;
}
