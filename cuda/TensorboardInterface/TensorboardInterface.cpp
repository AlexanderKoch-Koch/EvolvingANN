#include "TensorboardInterface.h"
#include <tensorflow/c/c_api.h>
#include <tensorflow/core/util/events_writer.h>
#include <tensorflow/core/lib/histogram/histogram.h>
#include <string>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unistd.h>
#include <dirent.h>

tensorflow::EventsWriter *writer;

void init_events_writer(const char *log_dir_arg)
{
    //find dir name which does not yet exist
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
    
    
    //delete old log dir if it exists
    DIR* dp;
    struct dirent* ep;
    char int_char[10];
    for(int i = -1; i< 10; i++){
      // iterate over some possible folder names
      sprintf(int_char, "%d", i);
      char path[128] = "";
      strcat(path, log_dir_arg);
      if(i != -1) strcat(path, int_char);
      std::string path_string = std::string(log_dir_arg);
      dp = opendir(path);
      
      if (dp != NULL)
      {
        // iterate over all files in the folder
        while(ep = readdir(dp))
        {
          // delete all files in diretcory
          char full_path[256] = "";
          strcat(full_path, path);
          strcat(full_path, "/");
          strcat(full_path, ep->d_name);
          remove(full_path);
        }
      }
      // remove directory if it exists
      rmdir(path);
    }
    
    //create log dir
    mkdir(log_dir, 0700);
    
    //initialize tensorboard writer object
    writer = new tensorflow::EventsWriter(strcat(log_dir, "/events"));
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
