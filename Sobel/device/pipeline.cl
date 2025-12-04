#pragma OPENCL EXTENSION cl_intel_channels : enable

#define TILE_SIZE 128
#define WIDTH 1920
#define HEIGHT 1080
#define TILES_W (WIDTH / TILE_SIZE)
#define TILES_H (HEIGHT / TILE_SIZE)
#define PACKET_SIZE 3
#define PACKET_H 3
#define PACKET_W 12
#define PIXEL_PER_PACKET 8
#define MEMORY_WIDTH 16

typedef struct {
   uchar data[PACKET_H][PACKET_W];
} channel_packet;

typedef struct {
   uchar data[PIXEL_PER_PACKET];
} out_packet;

channel channel_packet sobel_channel __attribute__((depth(32)));
channel out_packet out_channel __attribute__((depth(32)));

__attribute__((max_global_work_dim(0)))
__kernel void mem_read(__global const uchar16* restrict input, int width, int height) {
    

    __local uchar local_tile[TILE_SIZE+2][TILE_SIZE+2];
    
    for (int height_count = 0; height_count < TILES_H; height_count++) {
        for (int width_count = 0; width_count < TILES_W; width_count++) {

            for (int h = 0; h < TILE_SIZE+2; h++){
                for (int w = 0; w < TILE_SIZE+2; w+=MEMORY_WIDTH){

                    int global_h = height_count * TILE_SIZE + h - 1;
                    int global_w_start = width_count * TILE_SIZE + w - 1;
                    
                    uchar16 final_data_vec; 

                    if (global_h < 0 || global_h >= height) {
                        final_data_vec = (uchar16)(0);
                    } else {
                        int width_in_vectors = width / MEMORY_WIDTH;
                        int vec_index_col = max(0,min(global_w_start/ MEMORY_WIDTH,width_in_vectors-1));
                        int vec_index = global_h * width_in_vectors + vec_index_col;
                        uchar16 raw_data_vec = input[vec_index];

                        #pragma unroll
                        for(int i = 0; i < MEMORY_WIDTH; ++i) {
                            int current_global_w = global_w_start + i;
                            final_data_vec[i] =
                                (current_global_w < 0 || current_global_w >= width) ? 0 : raw_data_vec[i];
                        }
                    }

                    #pragma unroll
                    for (int i = 0; i < MEMORY_WIDTH; i++) {
                        if ((w + i) < TILE_SIZE + 2) {
                            local_tile[h][w + i] = final_data_vec[i];
                        }
                    }

                    

                }
            }

            for (int h = 0; h < TILE_SIZE; h++){
                for (int w = 0; w < TILE_SIZE; w+=PIXEL_PER_PACKET){
                    
                    channel_packet packet;
                    
                    #pragma unroll
                    for (int i = 0; i < PACKET_H; i++){
                        #pragma unroll
                        for (int j = 0; j < PACKET_W; j++){
                            
                            packet.data[i][j] = local_tile[h+i][w+j];

                        }
                    }

                    write_channel_intel(sobel_channel, packet);
                }
            }

            
            
        }
    }

}


__attribute__((max_global_work_dim(0)))
__kernel void sobel(
    int width,
    int height
) {

    const char sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    //const char sobel_y[3][3] = {{ 1, 2, 1}, { 0, 0, 0}, {-1,-2,-1}};

    const int total_pixels_to_process = ((TILES_H * TILE_SIZE) * (TILES_W * TILE_SIZE))/PIXEL_PER_PACKET;

    for (int t = 0; t < total_pixels_to_process; t++) {

        channel_packet packet = read_channel_intel(sobel_channel);

        out_packet salida;

        #pragma unroll
        for (int w = 0; w < PIXEL_PER_PACKET; w++){
            int gx = 0;
            //int gy = 0;
            #pragma unroll
            for (int i = 0; i < 3; i++){
                #pragma unroll
                for (int j = 0; j < 3; j++){
                    
                    uchar pixel = packet.data[i][j + w];
                    gx += pixel * sobel_x[i][j];
                    //gy += pixel * sobel_y[i][j];

                }
            }

            //int magnitude = (int)sqrt((float)(gx * gx + gy * gy));
            int magnitude = abs(gx);
            if (magnitude > 255) magnitude = 255;
            
            salida.data[w] = (uchar)magnitude;

        }
        
        write_channel_intel(out_channel, salida);
        
    }

}

__attribute__((max_global_work_dim(0)))
__kernel void mem_write(
    __global uchar16* restrict output,
    int width,
    int height
) {
    const int width_in_vectors = width / MEMORY_WIDTH;
    
    const int PACKETS_PER_WRITE = MEMORY_WIDTH / PIXEL_PER_PACKET;

    for (int height_count = 0; height_count < TILES_H; height_count++) {
        for (int width_count = 0; width_count < TILES_W; width_count++) {
            for (int h = 0; h < TILE_SIZE; h++) {
                for (int w = 0; w < TILE_SIZE; w += MEMORY_WIDTH) {
                    
                    uchar16 write_vector;

                    for (int pkt_idx = 0; pkt_idx < PACKETS_PER_WRITE; pkt_idx++) {
                        
                        out_packet received_packet = read_channel_intel(out_channel);
                        
                        #pragma unroll
                        for (int i = 0; i < PIXEL_PER_PACKET; i++) {
                            int dest_index = pkt_idx * PIXEL_PER_PACKET + i;
                            write_vector[dest_index] = received_packet.data[i];
                        }
                    }

                    int global_h = height_count * TILE_SIZE + h;
                    int global_w = width_count * TILE_SIZE + w;

                    if (global_h < height && global_w < width) {
                        int vec_index = global_h * width_in_vectors + (global_w / MEMORY_WIDTH);
                        output[vec_index] = write_vector;
                    }
                }
            }
        }
    }
}
