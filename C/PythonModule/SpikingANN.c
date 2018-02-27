#include <Python.h>
#include "Brain.h"
#include "Parameters.h"
#include <numpy/arrayobject.h>



static PyObject *spikingann_init(PyObject *self, PyObject *args)
{
    int num_neurons, num_inputs, num_outputs;

    if (!PyArg_ParseTuple(args, "iii", &num_neurons, &num_inputs, &num_outputs))
    {
      printf("This function takes 3 integer arguments (num_neurons, num_inputs, num_outputs)");
      return NULL;
    }
    init_brain(num_neurons, num_inputs, num_outputs);
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
  int *brain_input;

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
  brain_input = (int*) malloc(sizeof(int) * len);
  for(int element = 0; element < len; element++){
    PyObject *next = PyIter_Next(iter);
    brain_input[element] = PyObject_IsTrue(next);
  }

  int output_length;
  int *outputs = think(brain_input, len, &output_length);
  printf("output_length %d\n", output_length);
  free(brain_input);

  PyObject *output_list = PyList_New(output_length);
  for (int i = 0; i < output_length; i++) {
    PyList_SET_ITEM(output_list, i, PyLong_FromLong((long)outputs[i]));
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
