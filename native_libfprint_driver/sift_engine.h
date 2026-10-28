#ifndef SIFT_ENGINE_H
#define SIFT_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*sift_enroll_progress_cb)(int completed_stages, void* user_data);

int sift_engine_init(void);
int sift_engine_enroll(unsigned char** out_data, sift_enroll_progress_cb progress_cb, void* user_data);
int sift_engine_verify(const unsigned char* saved_data, int data_size);

#ifdef __cplusplus
}
#endif

#endif // SIFT_ENGINE_H