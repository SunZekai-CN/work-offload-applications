#pragma once

#include <OVR_CAPI.h>
#include <OVR_CAPI_GL.h>
#include <Extras/OVR_Math.h>

#include "lib/Rectangle2D.h"
#include "lib/Shader.h"

#include "OutputContext.h"

class RiftOutput : public OutputContext
{
public:
    RiftOutput(cv::Size backbufferSize, uint cameraWidth, uint cameraHeight, float cameraFovH, bool invertColour);
    ~RiftOutput();

    void newFrame(int& frameIndex, ovrPosef poses[2]);
    void setFramePose(int frameIndex, ovrPosef poses[2]);

    void renderScene(Renderer* ctx, int hit) override;

private:
    ovrSession mSession;
    ovrGraphicsLuid mLuid;
    ovrHmdDesc mHmdDesc;
    ovrSizei mBufferSize;
    ovrTextureSwapChain mTextureChain;
    GLuint mFramebufferId;
    GLuint mDepthBufferId;
    ovrMirrorTexture mMirrorTexture;
    GLuint mMirrorTextureId;
    ovrVector3f mHmdToEyeOffset[2];

    int mFrameIndex;
    ovrPosef mEyePose[2];

    cv::Size mFrameSize;

    unique_ptr<Rectangle2D> mFullscreenQuad;
    shared_ptr<Shader> mRiftMirrorShader;

};