#include <Python.h>
#include "Brain.h"
#include "Parameters.h"



static PyObject *spikingann_init(PyObject *self, PyObject *args)
{
    int num_neurons;
    int num_inputs;
    int num_outputs;

    if (!PyArg_ParseTuple(args, "iii", &num_neurons, &num_inputs, &num_outputs))
    {
        return NULL;
    }
    printf("jnum outputs %d", num_outputs);
    printf("num neurons %d\n", num_neurons);
    init_brain(num_neurons, num_inputs, num_outputs);
    printf("%d calls",get_num());
    printf("mojn");

    Py_RETURN_NONE;
}

static PyMethodDef GreetMethods[] = {
    {"init", spikingann_init, METH_VARARGS, "intitialize brain"},
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
