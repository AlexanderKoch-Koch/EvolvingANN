#include "TensorboardInterface.h"
#include <tensorflow/c/c_api.h>
#include <tensorflow/core/util/events_writer.h>
#include <string>
#include <iostream>

void write_scalar(int scalar)
{   
    printf("creating event file");
    std::string envent_file = "./events";
    tensorflow::EventsWriter writer(envent_file);
    for (int i = 0; i < 150; ++i){
        printf("writing value %i", i);
        tensorflow::Event event;
        event.set_wall_time(i*20);
        event.set_step(i);
        tensorflow::Summary::Value* summ_val = event.mutable_summary()->add_value();
        summ_val->set_tag("scalar");
        summ_val->set_simple_value(scalar);
        writer.WriteEvent(event);
    }
}


