#include <stdint.h>
#define LAB_ABI 2
#define RIGHT 1u
#define LEFT 2u
#define DOWN 4u
#define UP 8u
#define START 16u
#define RESET 32u
#define BACK 64u
void rect(int32_t,int32_t,int32_t,int32_t,int32_t);
int32_t audio_state(void);int32_t audio_position(void);int32_t audio_record(int32_t);
int32_t audio_classify(void);int32_t audio_reply(int32_t);int32_t audio_stop(void);
