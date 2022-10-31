#include "Common.h"

#include "lib/Rectangle2D.h"
#include "lib/Shader.h"

#include "camera/RealsenseCamera.h"
#include "camera/ZEDCamera.h"

class StereoCalibrate : public App
{
public:
    StereoCalibrate()
    {
    }

    void init() override
    {
        mZed = new ZEDCamera(sl::zed::VGA, 60);
        mRealsense = new RealsenseCamera(640, 480, 60, RealsenseCamera::ENABLE_COLOUR);

        // Create OpenGL images to visualise the calibration
        glGenTextures(2, mTexture);
        glBindTexture(GL_TEXTURE_2D, mTexture[0]);
        GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
            mRealsense->getWidth(RealsenseCamera::COLOUR), mRealsense->getHeight(RealsenseCamera::COLOUR),
            0, GL_BGR, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, mTexture[1]);
        GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
            mZed->getWidth(ZEDCamera::LEFT), mZed->getHeight(ZEDCamera::LEFT),
            0, GL_BGR, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    ~StereoCalibrate()
    {
        delete mZed;
        delete mRealsense;
    }

    void render() override
    {
        // Read the left frame from both cameras
        mRealsense->capture();
        mRealsense->copyData();
        mRealsense->copyFrameIntoCVImage(RealsenseCamera::COLOUR, &mFrame[0]);
        mZed->capture();
        mZed->copyData();
        mZed->copyFrameIntoCVImage(ZEDCamera::LEFT, &mFrame[1]);

        // Display them
        static Rectangle2D leftQuad(glm::vec2(0.0f, 0.0f), glm::vec2(0.5f, 1.0f));
        static Rectangle2D rightQuad(glm::vec2(0.5f, 0.0f), glm::vec2(1.0f, 1.0f));
        static Shader shader("../media/quad.vs", "../media/quad.fs");
        shader.bind();
        glBindTexture(GL_TEXTURE_2D, mTexture[0]);
        GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
            mRealsense->getWidth(RealsenseCamera::COLOUR), mRealsense->getHeight(RealsenseCamera::COLOUR),
            GL_BGR, GL_UNSIGNED_BYTE, mFrame[0].ptr()));
        leftQuad.render();
        glBindTexture(GL_TEXTURE_2D, mTexture[1]);
        GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
            mZed->getWidth(ZEDCamera::LEFT), mZed->getHeight(ZEDCamera::LEFT),
            GL_BGR, GL_UNSIGNED_BYTE, mFrame[1].ptr()));
        rightQuad.render();
    }

    void keyEvent(int key, int scancode, int action, int mods) override
    {
        static cv::Size boardSize(9, 6);
        static double squareSize = 0.025; // 25mm

        if (action == GLFW_PRESS)
        {
            if (key == GLFW_KEY_SPACE)
            {
                // Find chessboard corners from both cameras
                cv::Mat greyscale;
                static std::vector<cv::Point2f> leftCorners, rightCorners;
                cvtColor(mFrame[0], greyscale, cv::COLOR_BGR2GRAY);
                bool leftValid = findChessboardCorners(greyscale, boardSize, leftCorners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
                cvtColor(mFrame[1], greyscale, cv::COLOR_BGR2GRAY);
                bool rightValid = findChessboardCorners(greyscale, boardSize, rightCorners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

                // If neither are valid, skip this frame
                if (!leftValid || !rightValid)
                    return;

                // Both are valid, add the pair of points
                mLeftCorners.push_back(leftCorners);
                mRightCorners.push_back(rightCorners);
                leftCorners.clear();
                rightCorners.clear();

                // Mark that a corner was added
                cout << "Captured a pair of corners - #" << mLeftCorners.size() << endl;
            }
            else if (key == GLFW_KEY_ENTER)
            {
                // We are done capturing pairs of corners, find extrinsic parameters
                std::vector<std::vector<cv::Point3f>> objectPoints;
                objectPoints.resize(mLeftCorners.size());
                for (int i = 0; i < mLeftCorners.size(); i++)
                {
                    for (int j = 0; j < boardSize.height; j++)
                        for (int k = 0; k < boardSize.width; k++)
                            objectPoints[i].push_back(cv::Point3f(j * squareSize, k * squareSize, 0.0));
                }

                CameraIntrinsics& zedIntr = mZed->getIntrinsics(ZEDCamera::LEFT);
                CameraIntrinsics& rsIntr = mRealsense->getIntrinsics(RealsenseCamera::COLOUR);

                cv::Mat R, T, E, F;
                double rms = stereoCalibrate(objectPoints, mLeftCorners, mRightCorners,
                    rsIntr.cameraMatrix, rsIntr.coeffs,
                    zedIntr.cameraMatrix, zedIntr.coeffs,
                    mFrame[0].size(), R, T, E, F,
                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5),
                    cv::CALIB_FIX_INTRINSIC);
                cout << "Stereo Calibration done with RMS error = " << rms << endl;
                cout << "R: " << R << endl;
                cout << "T: " << T << endl;
                cout << "E: " << E << endl;
                cout << "F: " << F << endl;

                // Save to a file
                cv::FileStorage fs("../stereo-params.xml", cv::FileStorage::WRITE);
                if (fs.isOpened())
                {
                    fs <<
                        "R" << R <<
                        "T" << T <<
                        "E" << E <<
                        "F" << F;
                    fs.release();
                }
                else
                {
                    THROW_ERROR("Unable to save calibration parameters");
                }
            }
        }
    }

    cv::Size getSize() override
    {
        return cv::Size(1280, 480);
    }

private:
    ZEDCamera* mZed;
    RealsenseCamera* mRealsense;

    GLuint mTexture[2];
    cv::Mat mFrame[2];
    std::vector<std::vector<cv::Point2f>> mLeftCorners;
    std::vector<std::vector<cv::Point2f>> mRightCorners;
};

DEFINE_MAIN(StereoCalibrate);
