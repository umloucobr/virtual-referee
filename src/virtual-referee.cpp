#include "virtual-referee.h"

int main(int argc, char* argv[]) {
	cv::namedWindow("Line Detector", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Line Detector 2", cv::WINDOW_AUTOSIZE);

	cv::VideoCapture video;
	video.open("c.mp4");

	cv::Mat vid1;
	cv::Mat vid2;
	cv::Mat vid2Gray;
	cv::Mat vid2Canny;
	cv::Mat pMOG2Mask;
	cv::Ptr<cv::BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2(500, 16.0, false);
	std::vector<cv::Vec2f> lines;
	std::vector<cv::Vec3f> circles;
	bool pause {false};

	while (!pause)
	{
		video >> vid1;
		if (vid1.empty())
		{
			return -1;
		}

		cv::GaussianBlur(vid1, vid2, cv::Size(5, 5), 5, 5);
		cv::cvtColor(vid2, vid2Gray, cv::COLOR_BGR2GRAY);

		cv::Canny(vid2Gray, vid2Canny, 50, 200, 3, true);
		cv::HoughLines(vid2Canny, lines, 1, CV_PI / 180, 150, 0, 0);

		pMOG2->apply(vid2, pMOG2Mask);
		cv::cvtColor(pMOG2Mask, pMOG2Mask, cv::COLOR_GRAY2BGR);

		for (int i = 0; i < lines.size(); i++)
		{
			float rho = lines[i][0];
			float theta = lines[i][1];

			cv::Point pt1;
			cv::Point pt2;

			double a = cos(theta);
			double b = sin(theta);
			double x0 = a * rho;
			double y0 = b * rho;

			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));

			cv::line(vid2, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
			cv::line(pMOG2Mask, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
		}

		cv::HoughCircles(vid2Canny, circles, cv::HOUGH_GRADIENT, 2.0, vid2Canny.cols / 10);

		for (size_t i = 0; i < circles.size(); ++i) {
			cv::circle(
				vid2,
				cv::Point(cvRound(circles[i][0]), cvRound(circles[i][1])),
				cvRound(circles[i][2]),
				cv::Scalar(0, 0, 255),
				2
			);
		} 

		cv::imshow("Line Detector", pMOG2Mask);
		cv::imshow("Line Detector 2", vid2);

		int ascii {static_cast<int>(cv::waitKey(33))};

		if (ascii == 112)
		{
			while (cv::waitKey(0) != 112)
			{
				cv::waitKey(0);
			}
		}
		else if (ascii == 27)
		{
			cv::destroyAllWindows();
			return 0;
		}
	}
	return 0;
}
