#include "TensorboardInterface.h"
#include <tensorflow/c/c_api.h>
#include <tensorflow/core/util/events_writer.h>
#include <tensorflow/core/lib/histogram/histogram.h>
#include <string>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

tensorflow::EventsWriter *writer;

int mkpath(char* file_path, mode_t mode) {
    // create path file_path with mode mode
  assert(file_path && *file_path);
  char* p;
  for (p=strchr(file_path+1, '/'); p; p=strchr(p+1, '/')) {
    *p='\0';
    if (mkdir(file_path, mode)==-1) {
      if (errno!=EEXIST) { *p='/'; return -1; }
    }
    *p='/';
  }
  return 0;
}

void init_events_writer(const char *log_dir_arg)
{
    // delete old log directory if exists
    std::string log_dir_string = std::string(log_dir_arg);
    std::string command = "rm -r " + log_dir_string;
    system(command.c_str());
    
    //create new empty log diretcory
    mkpath((char*) (log_dir_string + "/").c_str(), 0700);
    
    //initialize tensorboard writer object
    writer = new tensorflow::EventsWriter(log_dir_string + "/events");
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


void write_histogram(unsigned long step_arg,  const char *tag) {
    for (int step = 0; step <27; step++){
        
        tensorflow::histogram::Histogram h;
        for (double  i = 0; i < step; i++)
          h.Add(i);
         
        printf("\nmedian: %f", h.Median());
        printf(" StandardDeviation: %f", h.StandardDeviation());
        // convert to proto
        tensorflow::HistogramProto *hist_proto = new tensorflow::HistogramProto();
        h.EncodeToProto(hist_proto, true);
        
        tensorflow::Event event;
        event.set_wall_time(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
        event.set_step(step);
        tensorflow::Summary::Value* summ_val = event.mutable_summary()->add_value();
        summ_val->set_tag(tag);
        summ_val->set_allocated_histo(hist_proto);
        writer->WriteEvent(event);
    
    }
}
