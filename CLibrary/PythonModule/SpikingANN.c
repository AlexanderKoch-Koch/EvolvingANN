#include <Python.h>
#include "Brain.h"
#include "Parameters.h"



static PyObject *spikingann_init(PyObject *self, PyObject *args)
{
  PyObject *parameter_dict;

    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &parameter_dict))
    {
      printf("This function takes a parameter dictionary");
      return NULL;
    }
  struct Parameters parameters;
  parameters.num_inputs = 5;
  parameters.num_inputs = PyLong_AsLong(PyDict_GetItem(parameter_dict,Py_BuildValue("s", "num_inputs")));
  parameters.num_neurons = PyLong_AsLong(PyDict_GetItem(parameter_dict,Py_BuildValue("s", "num_neurons")));
  parameters.num_outputs = PyLong_AsLong(PyDict_GetItem(parameter_dict,Py_BuildValue("s", "num_outputs")));
  parameters.num_synapses_per_neuron = PyLong_AsLong(PyDict_GetItem(parameter_dict,Py_BuildValue("s", "num_synapses_per_neuron")));
  parameters.learning_rate = PyFloat_AsDouble(PyDict_GetItem(parameter_dict,Py_BuildValue("s", "learning_rate")));
  parameters.threshold = PyFloat_AsDouble(PyDict_GetItem(parameter_dict,Py_BuildValue("s", "threshold")));
  parameters.activity_discount_factor = PyFloat_AsDouble(PyDict_GetItem(parameter_dict,Py_BuildValue("s", "activity_discount_factor")));
  parameters.max_weight_value = PyFloat_AsDouble(PyDict_GetItem(parameter_dict,Py_BuildValue("s", "max_weight_value")));
  parameters.max_start_weight_sum = PyFloat_AsDouble(PyDict_GetItem(parameter_dict,Py_BuildValue("s", "max_start_weight_sum")));
  parameters.min_weight = PyFloat_AsDouble(PyDict_GetItem(parameter_dict,Py_BuildValue("s", "min_weight")));
  init_brain(parameters);
  Py_RETURN_NONE;
}

static PyObject *spikingann_reward(PyObject *self, PyObject *args)
{
    float reward;

    if (!PyArg_ParseTuple(args, "f", &reward))
    {
      printf("This function takes 1 integer argument reward)");
      return NULL;
    }
    process_reward(reward);
    Py_RETURN_NONE;
}

static PyObject *spikingann_think(PyObject *self, PyObject *args)
{
  PyObject *obj;
  float *brain_input;

  if (!PyArg_ParseTuple(args, "O", &obj)) {
    printf("argument has to be exactly one list");
    return NULL;
  }

  PyObject *iter = PyObject_GetIter(obj);
  if (!iter) {
    printf("argument has to be a list");
    return NULL;
  }

  int len = Py_SAFE_DOWNCAST(PyObject_Size(obj), Py_ssize_t, int);
  brain_input = (float*) malloc(sizeof(float) * len);
  for(int element = 0; element < len; element++){
    PyObject *next = PyIter_Next(iter);
    brain_input[element] = PyFloat_AsDouble(next);
  }

  int output_length;
  int *outputs = think(brain_input, len, &output_length);
  //printf("output_length %d\n", output_length);
  free(brain_input);

  PyObject *output_list = PyList_New(output_length);
  for (int i = 0; i < output_length; i++) {
    PyList_SET_ITEM(output_list, i, PyFloat_FromDouble((double)outputs[i]));
  }

  return output_list;
}

static PyObject *spikingann_release_memory(PyObject *self, PyObject *args)
{
    release_memory();
    Py_RETURN_NONE;
}

static PyObject *spikingann_reset_memory(PyObject *self, PyObject *args)
{
    reset_memory();
    Py_RETURN_NONE;
}

static PyMethodDef GreetMethods[] = {
    {"init", spikingann_init, METH_VARARGS, "intitialize brain"},
    {"reward", spikingann_reward, METH_VARARGS, "reward brain"},
    {"think", spikingann_think, METH_VARARGS, "think"},
    {"reset_memory", spikingann_reset_memory, METH_VARARGS, "reset memory"},
    {"release_memory", spikingann_release_memory, METH_VARARGS, "free all allocated memory"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef spikingannmodule = {
    PyModuleDef_HEAD_INIT,
    "spikingann", /* module name */
    NULL, /* module documentation, may be NULL */
    -1,
    GreetMethods /* the methods array */
};


PyMODINIT_FUNC PyInit_spikingann(void)
{
    return PyModule_Create(&spikingannmodule);
}
