#include "virtual-referee.h"

//Fixed size queue.
namespace vr {
	template <typename T, int MaxLen, typename Container = std::deque<T>>
	class FixedQueue : public std::queue<T, Container> {
	public:
		void push(const T& value) {
			if (this->size() == MaxLen) {
				this->c.pop_front();
			}
			std::queue<T, Container>::push(value);
		}
	};
}

void writeOnDisk(vr::FixedQueue<std::string, 120>& buffer, int fps, const cv::Size& size, bool& alreadySaved) {
	std::string path{};
	if (alreadySaved)
	{
		path = "lastseconds2.mp4";
	}
	else
	{
		path = "lastseconds1.mp4";
		alreadySaved = true;
	}
	
	cv::VideoWriter writer;
	writer.open(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size, true);
	
	while (!buffer.empty())
	{
		cv::Mat bufferFrame(cv::imread(buffer.front()));
		writer.write(bufferFrame);
		buffer.pop();
	}
}

int main(int argc, char* argv[]) {
	int fps {30};
	bool useCircles {false};
	bool useFindContours {false};
	cv::String pathToVideo {};

	if (argc > 1)
	{
		const cv::String keys{
			"{help h usage ? |      | Print this message}"
			"{@path			 |<none>| Path of the video/camera}"
			"{ci circle		 |      | Use HoughCircles to find circles (expensive)}" 
			"{co contours	 |      | Use FindContour to find contours (expensive)}"};

		cv::CommandLineParser parser(argc, argv, keys);
		parser.about("Virtual Referee V0.8");

		pathToVideo = parser.get<cv::String>(0);

		if (parser.has("help"))
		{
			parser.printMessage();
			return 0;
		}

		if (parser.has("ci"))
		{
			useCircles = true;
		}

		if (parser.has("co"))
		{
			useFindContours = true;
		}
	}

	cv::namedWindow("Line Detector", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Line Detector 2", cv::WINDOW_AUTOSIZE);

	cv::VideoCapture video;
	video.open(pathToVideo);

	cv::Size size(
		static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH)),
		static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT))
	);

	if (!video.isOpened())
	{
		std::cerr << "Could not open video";
		return -1;
	}

	cv::Mat vid1; //First frame.
	cv::Mat vid2;	//Frame with gaussian blur.
	cv::Mat vid2Gray;
	cv::Mat vid2Canny;
	cv::Mat pMOG2Mask;
	cv::Ptr<cv::BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2(500, 16.0, false);
	std::vector<cv::Vec2f> lines;	//Find lines.
	std::vector<cv::Vec3f> circles;	//Find circles.
	std::vector<std::vector<cv::Point>> contours;	//Find contours.
	std::vector<cv::Vec4i> hierarchy;				//Find contours.

	//Video buffer that gets last seconds of video. It has 900 frames for 30 FPS and 1800 for 60 FPS (because the maximum is 30 seconds).
	int currentFrame {0};
	bool alreadySaved {false}; //This bool determines if it is the first or second video buffer.
	std::string videoBufferFolderHo {"videobuffer/hl/vb"}; //Video buffer for frame vid2.
	std::string videoBufferFolderBa {"videobuffer/ba/vb"}; //Video buffer for frame Background Subtractor.
	vr::FixedQueue<std::string, 120>imagePathHo;
	vr::FixedQueue<std::string, 120>imagePathBa;

	while (true)
	{
		video >> vid1;
		if (vid1.empty())
		{
			return -1;
		}

		++currentFrame;

		std::string videoBufferPathHo{ videoBufferFolderHo + std::to_string(currentFrame) + ".jpg" };
		std::string videoBufferPathBa{ videoBufferFolderBa + std::to_string(currentFrame) + ".jpg" };

		cv::GaussianBlur(vid1, vid2, cv::Size(5, 5), 5, 5);
		cv::cvtColor(vid2, vid2Gray, cv::COLOR_BGR2GRAY);

		cv::Canny(vid2Gray, vid2Canny, 50, 200, 3, true);
		cv::HoughLines(vid2Canny, lines, 1, CV_PI / 180, 200, 0, 0);

		//Apply background subtractor.
		pMOG2->apply(vid2, pMOG2Mask);

		if (useCircles)
		{
			cv::HoughCircles(vid2Gray, circles, cv::HOUGH_GRADIENT, 1.5, 400, 100.0, 100.0, 80, 250);

			for (size_t i = 0; i < circles.size(); ++i) {
				cv::circle(
					vid2,
					cv::Point(cvRound(circles[i][0]), cvRound(circles[i][1])),
					cvRound(circles[i][2]),
					cv::Scalar(0, 0, 255),
					2
				);
			}
		}

		if (useFindContours)
		{
			findContours(pMOG2Mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

			cv::Mat drawing = cv::Mat::zeros(pMOG2Mask.size(), CV_8UC3);
			for (size_t i = 0; i < contours.size(); i++)
			{
				cv::Scalar color = cv::Scalar(10, 10, 10);
				drawContours(vid2, contours, static_cast<int>(i), color, 2, cv::LINE_8, hierarchy, 0);
			}
		}

		//Convert back to color to draw a colored line with the data that HoughLine() produced.
		cv::cvtColor(pMOG2Mask, pMOG2Mask, cv::COLOR_GRAY2BGR);

		//Draw HoughLines() to both images.
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

		cv::imshow("Line Detector", pMOG2Mask);
		cv::imshow("Line Detector 2", vid2);

		if (currentFrame == 120)
		{
			currentFrame = 0;
		}
		imagePathHo.push(videoBufferPathHo);
		cv::imwrite(videoBufferPathHo, vid2);

		imagePathBa.push(videoBufferPathBa);
		cv::imwrite(videoBufferPathBa, pMOG2Mask);

		//ASCII 112 = "p"
		//ASCII 114 = "r"
		//ASCII 27 = "ESC".
		int ascii {static_cast<int>(cv::waitKey(1))};
		if (ascii == 112)
		{
			while (cv::waitKey(0) != 112)
			{
				cv::waitKey(0);
			}
		}
		else if (ascii == 114)
		{
			writeOnDisk(imagePathHo, fps, size, alreadySaved);
			writeOnDisk(imagePathBa, fps, size, alreadySaved);
		}
		else if (ascii == 27)
		{
			cv::destroyAllWindows();
			return 0;
		}
	}
	video.release();
	return 0;
}
