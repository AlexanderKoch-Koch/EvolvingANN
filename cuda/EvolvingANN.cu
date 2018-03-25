#include <Python.h>
#include "Brain.h"


static PyObject *eann_init(PyObject *self, PyObject *args)
{
  int number;

    if (!PyArg_ParseTuple(args, "i", &number))
    {
      printf("This function takes an integer");
      return NULL;
    }
  printf("your number is %d", number);
  init();
  Py_RETURN_NONE;
}

static PyObject *eann_reward(PyObject *self, PyObject *args)
{
    float reward;

    if (!PyArg_ParseTuple(args, "f", &reward))
    {
      printf("This function takes 1 integer argument reward)");
      return NULL;
    }
    //process_reward(reward);
    Py_RETURN_NONE;
}



static PyObject *eann_release_memory(PyObject *self, PyObject *args)
{
    //release_memory();
    Py_RETURN_NONE;
}

static PyObject *eann_reset_memory(PyObject *self, PyObject *args)
{
    //reset_memory();
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"init", eann_init, METH_VARARGS, "intitialize brain"},
    {"reward", eann_reward, METH_VARARGS, "reward brain"},
    {"reset_memory", eann_reset_memory, METH_VARARGS, "reset memory"},
    {"release_memory", eann_release_memory, METH_VARARGS, "free all allocated memory"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef eann_module = {
    PyModuleDef_HEAD_INIT,
    "eann", /* module name */
    NULL, /* module documentation, may be NULL */
    -1,
    methods /* the methods array */
};


PyMODINIT_FUNC PyInit_eann(void)
{
    return PyModule_Create(&eann_module);
}
