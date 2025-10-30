#pragma OPENCL EXTENSION cl_intel_channels : enable

#define PACKET_SIZE (3*3*3)
#define conv1_width 320
#define conv1_height 240
#define conv1_channels_in 3
#define conv1_out_width 160
#define conv1_out_height 120
#define conv1_channels_out 16
#define conv1_stride 2
#define conv1_x_zero 127
#define conv1_x_scale 0.0078125f
#define conv1_w_zero 151
#define conv1_w_scale 0.005215124227106571f
#define conv1_y_zero 0
#define conv1_y_scale 0.02971666119992733f

#define conv2_width 160
#define conv2_height 120
#define conv2_channels 16
#define conv2_x_zero 0
#define conv2_x_scale 0.02971666119992733
#define conv2_w_zero 146
#define conv2_w_scale 0.049183622002601624
#define conv2_y_zero 0
#define conv2_y_scale 0.04393049702048302

#define conv3_width 160
#define conv3_height 120
#define conv3_channels_in 16
#define conv3_channels_out 32
#define conv3_x_zero 0
#define conv3_x_scale 0.04393049702048302
#define conv3_w_zero 125
#define conv3_w_scale 0.008299560286104679
#define conv3_y_zero 0
#define conv3_y_scale 0.046722110360860825



#define w_size (conv1_channels_out * conv1_channels_in * 3 * 3)
#define MEM_WIDTH 16


typedef struct {
    uchar data[PACKET_SIZE];
} Packet;

channel Packet conv_channel __attribute__((depth(16)));
channel int requantize_channel __attribute__((depth(16)));
channel uchar conv1_out_channel __attribute__((depth(16)));
channel uchar conv2_out_channel __attribute__((depth(16)));
channel uchar pipe_out_channel __attribute__((depth(16)));

#define line_buffer_h 3 
#define line_buffer_w 336 

#define DATA_START_OFFSET 16

__attribute__((max_global_work_dim(0)))
__kernel void mem_read(__global const uchar* restrict x) {

    const int VECTORS_PER_LINE = conv1_width / MEM_WIDTH;
    __local uchar Line_buffer[conv1_channels_in * line_buffer_h * line_buffer_w];

    for (int y_out = 0; y_out < conv1_out_height; ++y_out) {
        
        int y_in_base = y_out * conv1_stride;

        #pragma unroll 1
        for (int kh = 0; kh < 3; ++kh) {
            int y_glob = y_in_base + kh - 1;
            #pragma unroll 1
            for (int c = 0; c < conv1_channels_in; ++c) {
                int local_line_offset = (c * line_buffer_h * line_buffer_w) + (kh * line_buffer_w);

                if (y_glob < 0 || y_glob >= conv1_height) {
                    #pragma unroll 1
                    for(int i = 0; i < line_buffer_w; ++i) {
                       Line_buffer[local_line_offset + i] = conv1_x_zero;
                    }
                } else {

                    Line_buffer[local_line_offset + DATA_START_OFFSET - 1] = conv1_x_zero; 
                    Line_buffer[local_line_offset + DATA_START_OFFSET + conv1_width] = conv1_x_zero; 

                    __global const uchar16 *p_glob = (__global const uchar16 *)(x + (c*conv1_height*conv1_width) + (y_glob*conv1_width));
                    __local uchar16 *p_loc  = (__local uchar16 *)(Line_buffer + local_line_offset + DATA_START_OFFSET);
                    
                    #pragma unroll 1
                    for (int i = 0; i < VECTORS_PER_LINE; ++i) {
                        p_loc[i] = p_glob[i];
                    }
                }
            }
        }

        for (int x_out = 0; x_out < conv1_out_width; ++x_out) {
            Packet paquete;
            int x_in_base = x_out * conv1_stride;

            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    #pragma unroll
                    for (int kc = 0; kc < conv1_channels_in; ++kc) {
                        int packet_idx = (kh * 3 + kw) * conv1_channels_in + kc;
                        int local_x = x_in_base + kw + (DATA_START_OFFSET - 1);
                        int local_idx = (kc * line_buffer_h * line_buffer_w) + (kh * line_buffer_w) + local_x;
                        paquete.data[packet_idx] = Line_buffer[local_idx];
                    }
                }
            }
            write_channel_intel(conv_channel, paquete);
        }
    }
}


#define M_INT_CONV1 45
#define SHIFT_CONV1 15

__attribute__((max_global_work_dim(0)))
__kernel void conv(
    __global const uchar* restrict weights,
    __global const int* restrict bias) 
{
    
    __local uchar local_w[w_size];
    __local int   local_b[conv1_channels_out];

    __global const uchar16* weights_vec = (__global const uchar16*)weights;
    __local uchar16* local_w_vec = (__local uchar16*)local_w;
    
    #pragma unroll 1
    for (int w = 0; w < w_size / MEM_WIDTH; w++){
        local_w_vec[w] = weights_vec[w];
    }

    __global const int4* bias_vec = (__global const int4*)bias;
    __local int4* local_b_vec = (__local int4*)local_b;
    
    #pragma unroll 1
    for (int b = 0; b < conv1_channels_out / 4; b++){
        local_b_vec[b] = bias_vec[b];
    }

    #pragma unroll 1
    for (int y_out = 0; y_out < conv1_out_height; ++y_out) {
        #pragma unroll 1
        for (int x_out = 0; x_out < conv1_out_width; ++x_out) {

            Packet paquete_in = read_channel_intel(conv_channel);
            
            #pragma unroll 1
            for (int cout = 0; cout < conv1_channels_out; cout += 2) {
                
                int resultados_paralelos[2];

                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    
                    int current_cout = cout + j;
                    int acc = local_b[current_cout];

                    #pragma unroll
                    for (int kh = 0; kh < 3; ++kh) {
                        #pragma unroll
                        for (int kw = 0; kw < 3; ++kw) {
                            #pragma unroll
                            for (int cin = 0; cin < conv1_channels_in; ++cin) {
                                
                                int packet_idx = (kh * 3 + kw) * conv1_channels_in + cin;
                                uchar val_in = paquete_in.data[packet_idx];

                                int weight_idx = (current_cout * (3*3*conv1_channels_in)) + packet_idx;
                                uchar val_w = local_w[weight_idx];

                                short a = (int)val_in - (int)conv1_x_zero;
                                short b = (int)val_w - (int)conv1_w_zero;

                                acc = mad24(a,b,acc);
                            }
                        }
                    }

                    long acc_mult = (long)acc * (long)M_INT_CONV1;
                    long acc_round = acc_mult + (1L << (SHIFT_CONV1 - 1));
                    int acc_scaled = (int)(acc_round >> SHIFT_CONV1);
                    int acc_final = acc_scaled + conv1_y_zero;
                    uchar valor_out = (uchar)clamp(acc_final, 0, 255);

                    resultados_paralelos[j] = valor_out;

                }

                for (int j = 0; j < 2; ++j) {
                    write_channel_intel(conv1_out_channel, resultados_paralelos[j]);
                }
            }
        }
    }
}

#define M_INT_DW 1090
#define SHIFT_DW 15

#define conv2_w_size (conv2_channels*3*3)

__attribute__((max_global_work_dim(0)))
__kernel void conv2(__global const uchar* restrict weights,
    __global const int* restrict bias)
{
    __local uchar local_w[conv2_w_size];
    __local int   local_b[conv2_channels];


    __global const uchar16* weights_vec = (__global const uchar16*)weights;
    __local uchar16* local_w_vec = (__local uchar16*)local_w;
    
    #pragma unroll 1
    for (int w = 0; w < conv2_w_size / MEM_WIDTH; w++){
        local_w_vec[w] = weights_vec[w];
    }


    __global const int4* bias_vec = (__global const int4*)bias;
    __local int4* local_b_vec = (__local int4*)local_b;
    
    #pragma unroll 1
    for (int b = 0; b < conv2_channels / 4; b++){
        local_b_vec[b] = bias_vec[b];
    }

    __local uchar line_buffer[conv2_channels][2][conv2_width];

    __local uchar window[conv2_channels][3][3];
    #pragma unroll 1
    for(int c=0; c<conv2_channels; ++c) {
        #pragma unroll 1
        for(int i=0; i<3; ++i) {
            #pragma unroll
            for(int j=0; j<3; ++j) {
                window[c][i][j] = conv2_x_zero;
            }
        }
    }

    
    for (int y = 0; y < conv2_height + 2; ++y) {
        for (int x = 0; x < conv2_width + 2; ++x) {
            
            int y_in = y - 1;
            int x_in = x - 1;

            #pragma unroll 1
            for (int c = 0; c < conv2_channels; ++c) {

                #pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
                    window[c][kh][0] = window[c][kh][1];
                    window[c][kh][1] = window[c][kh][2];
                }

                uchar new_pixel;
                if (y_in >= 0 && y_in < conv2_height && x_in >= 0 && x_in < conv2_width) {
                    new_pixel = read_channel_intel(conv1_out_channel);
                } else {
                    new_pixel = conv2_x_zero;
                }

                uchar pix_L0, pix_L1;
                if (x_in >= 0 && x_in < conv2_width) {
                    pix_L0 = line_buffer[c][(y+1)%2][x_in]; 
                    pix_L1 = line_buffer[c][y%2][x_in];
                } else {
                    pix_L0 = conv2_x_zero;
                    pix_L1 = conv2_x_zero;
                }

              
                window[c][0][2] = (y < 2) ? conv2_x_zero : pix_L0; 
                window[c][1][2] = (y < 1) ? conv2_x_zero : pix_L1; 
                window[c][2][2] = new_pixel;
                
                if (y_in >= 0 && y_in < conv2_height && x_in >= 0 && x_in < conv2_width) {
                    line_buffer[c][y%2][x_in] = new_pixel;
                }
            } 

          
            int y_out = y - 1;
            int x_out = x - 1;

            if (y_out >= 0 && y_out < conv2_height && x_out >= 0 && x_out < conv2_width) {
                
             
                #pragma unroll 1 
                for (int c = 0; c < conv2_channels; ++c) {
                    
                    int acc = local_b[c]; 
                    #pragma unroll
                    for (int kh = 0; kh < 3; ++kh) {
                        #pragma unroll
                        for (int kw = 0; kw < 3; ++kw) {
                            uchar val_in = window[c][kh][kw];
                            
                            int w_idx = (c * 9) + (kh * 3) + kw;
                            uchar val_w = local_w[w_idx];

                            short a = (int)val_in - conv2_x_zero;
                            short b = (int)val_w - conv2_w_zero;
                            acc = mad24(a, b, acc);
                        }
                    }

                    long acc_mult = (int)acc * (int)M_INT_DW;
                    long acc_round = acc_mult + (1L << (SHIFT_DW - 1));
                    long acc_scaled = (int)(acc_round >> SHIFT_DW);
                    long acc_final = acc_scaled + conv2_y_zero;
                    uchar valor_out = (uchar)clamp((int)acc_final, 0, 255);

                    write_channel_intel(conv2_out_channel, valor_out);
                }
            }
        } 
    } 

}

#define pw_w_size (conv3_channels_in*conv3_channels_out)
#define M_INT_PW 256
#define SHIFT_PW 15

__attribute__((max_global_work_dim(0)))
__kernel void conv3(__global const uchar* restrict weights,
    __global const int* restrict bias){


    __local uchar local_w[pw_w_size];

    __global const uchar16* weights_vec = (__global const uchar16*)weights;
    __local uchar16* local_w_vec = (__local uchar16*)local_w;
    #pragma unroll 1
    for (int w = 0; w < (pw_w_size + MEM_WIDTH - 1) / MEM_WIDTH; w++){
        local_w_vec[w] = weights_vec[w];
    }


    __local int local_b[conv3_channels_out];

    __global const int4* bias_vec = (__global const int4*)bias;
    __local int4* local_b_vec = (__local int4*)local_b;

    #pragma unroll 1
    for (int b = 0; b < conv3_channels_out / 4; b++){
        local_b_vec[b] = bias_vec[b];
    }
    

    uchar pixel_buffer[conv3_channels_in];

    #pragma unroll 1
    for (int y = 0; y < conv3_height; ++y) {
        #pragma unroll 1
        for (int x = 0; x < conv3_width; ++x) {

            #pragma unroll
            for (int c_in = 0; c_in < conv3_channels_in; ++c_in) {
                pixel_buffer[c_in] = read_channel_intel(conv2_out_channel);
            }

            #pragma unroll 1
            for (int cout = 0; cout < conv3_channels_out; cout += 2) {
                
                uchar resultados_paralelos[2];

                #pragma unroll 1
                for (int j = 0; j < 2; ++j) {
                    
                    int current_cout = cout + j;
                    int acc = local_b[current_cout];

                    #pragma unroll
                    for (int cin = 0; cin < conv3_channels_in; ++cin) {
                        
                        uchar val_in = pixel_buffer[cin];
                        
                        int weight_idx = (current_cout * conv3_channels_in) + cin;
                        uchar val_w = local_w[weight_idx];

                        short a = (int)val_in - conv3_x_zero;
                        short b = (int)val_w - conv3_w_zero;
                        acc = mad24(a, b, acc);
                    }

                    long acc_mult = (long)acc * (long)M_INT_PW;
                    long acc_round = acc_mult + (1L << (SHIFT_PW - 1));
                    int acc_scaled = (int)(acc_round >> SHIFT_PW);
                    int acc_final = acc_scaled + conv3_y_zero;
                    
                    resultados_paralelos[j] = (uchar)clamp(acc_final, 0, 255);
                }

                for (int j = 0; j < 2; ++j) {
                    write_channel_intel(pipe_out_channel, resultados_paralelos[j]);
                }
            }
        }
    }

}


#define VECTORS_PER_OUT_LINE (conv1_out_width / MEM_WIDTH)

__attribute__((max_global_work_dim(0)))
__kernel void mem_write(__global uchar* restrict y_out) {
    
    __local uchar reorder_buffer[conv1_out_width * conv3_channels_out];

    for (int y = 0; y < conv1_out_height; ++y) {

        for (int x = 0; x < conv1_out_width; ++x) {
            for (int c = 0; c < conv3_channels_out; ++c) {
                uchar val = read_channel_intel(pipe_out_channel);
                reorder_buffer[x * conv3_channels_out + c] = val;
            }
        }

        for (int c = 0; c < conv3_channels_out; ++c) {

            int global_base_addr = (c * conv1_out_height * conv1_out_width) + (y * conv1_out_width);
            __global uchar16 *p_glob = (__global uchar16 *)(y_out + global_base_addr);

            for (int i = 0; i < VECTORS_PER_OUT_LINE; ++i) {
                
                uchar16 data_vec; 

                #pragma unroll 1
                for (int j = 0; j < MEM_WIDTH; ++j) {
                    int current_x = i * MEM_WIDTH + j;
                    data_vec[j] = reorder_buffer[current_x * conv3_channels_out + c];
                }
                
                p_glob[i] = data_vec;
            }
        }
    }
}