#version 430

#define X C_X_
#define Y C_Y_
#define Z C_Z_

#define S C_S_

layout(local_size_x=X,local_size_y=Y,local_size_z=Z)in;
layout(std430,binding=0)buffer out_0
{
    float outxs[1];
};

layout(std430,binding=1)buffer in_1
{
    float a[1];
};
layout(std430,binding=2)buffer in_2
{
    float b[1];
};

#define win_width 5
#define win_height 5
#define win_wh 25

void main()
{
    // define consts
    const int x=int(gl_LocalInvocationID.x);
    const int y=int(gl_WorkGroupID.x);
    const int frag_i=x+y*X;
    outxs[frag_i]=a[frag_i]*b[frag_i];
}
