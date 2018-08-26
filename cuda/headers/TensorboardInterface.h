
#ifdef __cplusplus
extern "C" 
{
#endif

    void init_events_writer(const char *log_dir);
    void write_scalar(unsigned long step, float scalar, const char *tag);


#ifdef __cplusplus
}
#endif