#include "TensorboardInterface.h"
#include <tensorflow/c/c_api.h>
#include <tensorflow/core/util/events_writer.h>
#include <string>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

tensorflow::EventsWriter *writer;

void init_events_writer(const char *log_dir_arg)
{
    std::string log_dir_string = std::string(log_dir_arg);
    struct stat st = {0};
    
    char int_str[32];
    char log_dir[64] = "";
    int i = -1;
    do {
        i++;
        sprintf(int_str, "%d", i);
        log_dir[0] = '\0';
        strcat(log_dir, log_dir_arg);
        if(i != 0) strcat(log_dir, int_str);
    }while(stat(log_dir, &st) != -1);
    std::cout << "\ncreating log dir: " << log_dir;
    mkdir(log_dir, 0700);
    strcat(log_dir, "/events");
    writer = new tensorflow::EventsWriter(log_dir);
}


void write_scalar(unsigned long step, float scalar, const char *tag)
{   
    tensorflow::Event event;
    event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    event.set_step(step);
    tensorflow::Summary::Value* summ_val = event.mutable_summary()->add_value();
    summ_val->set_tag(tag);
    summ_val->set_simple_value(scalar);
    writer->WriteEvent(event);
    writer->Flush();
}


