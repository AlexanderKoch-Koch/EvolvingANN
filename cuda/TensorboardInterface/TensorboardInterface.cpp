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



/*int TensorBoardLogger::add_histogram(const string &tag, int step, vector<float> &values) {
    if (bucket_limits_ == NULL) {
        generate_default_buckets();
    }

    vector<int> counts(bucket_limits_->size(), 0);
    double min = numeric_limits<double>::max();
    double max = numeric_limits<double>::lowest();
    double sum = 0.0;
    double sum_squares = 0.0;
    for (auto v: values) {
        auto lb = lower_bound(bucket_limits_->begin(), bucket_limits_->end(), v);
        counts[lb - bucket_limits_->begin()]++;
        sum += v;
        sum_squares += v * v;
        if (v > max) {
            max = v;
        } else if (v < min) {
            min = v;
        }
    }

    auto histo = new HistogramProto();
    histo->set_min(min);
    histo->set_max(max);
    histo->set_num(values.size());
    histo->set_sum(sum);
    histo->set_sum_squares(sum_squares);
    for (size_t i = 0; i < counts.size(); ++i) {
        if (counts[i] > 0) {
            histo->add_bucket_limit((*bucket_limits_)[i]);
            histo->add_bucket(counts[i]);
        }
    }

    auto summary = new Summary();
    auto v = summary->add_value();
    v->set_node_name(tag);
    v->set_tag(tag);
    v->set_allocated_histo(histo);

    return add_event(step, summary);
}*/
