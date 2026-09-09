#define FP_COMPONENT "device_elan"

#include "drivers_api.h"
#include "sift_engine.h"
#include <glib.h>
#include <stdlib.h>
#include <string.h>

struct _FpiDeviceElan {
    FpDevice parent;
};

G_DECLARE_FINAL_TYPE (FpiDeviceElan, fpi_device_elan, FPI, DEVICE_ELAN, FpDevice);
G_DEFINE_TYPE (FpiDeviceElan, fpi_device_elan, FP_TYPE_DEVICE);

static void elan_open(FpDevice *device) {
    sift_engine_init();
    fpi_device_open_complete(device, NULL);
}

static void elan_close(FpDevice *device) {
    fpi_device_close_complete(device, NULL);
}

/* ==================== ENROLL ASYNC WORKER ==================== */

typedef struct {
    FpDevice *device;
    unsigned char *data;
    int size;
} EnrollTaskData;

static gboolean elan_enroll_complete_idle(gpointer user_data) {
    EnrollTaskData *task = (EnrollTaskData *)user_data;
    FpPrint *print = NULL;
    fpi_device_get_enroll_data(task->device, &print);

    if (task->size > 0 && print != NULL) {
        GVariant *record = g_variant_new_fixed_array(G_VARIANT_TYPE_BYTE, task->data, task->size, sizeof(guchar));
        g_object_set(print, "fpi-type", FPI_PRINT_RAW, "fpi-data", record, NULL);
        fpi_device_enroll_complete(task->device, g_object_ref(print), NULL);
    } else {
        fpi_device_enroll_complete(task->device, NULL, fpi_device_error_new(FP_DEVICE_ERROR_GENERAL));
    }

    if (task->data) free(task->data);
    g_free(task);
    return G_SOURCE_REMOVE;
}

static gpointer elan_enroll_worker_thread(gpointer user_data) {
    EnrollTaskData *task = (EnrollTaskData *)user_data;
    task->size = sift_engine_enroll(&task->data);
    g_idle_add(elan_enroll_complete_idle, task);
    return NULL;
}

static void elan_enroll(FpDevice *device) {
    EnrollTaskData *task = g_new0(EnrollTaskData, 1);
    task->device = device;
    GThread *thread = g_thread_new("elan-enroll-worker", elan_enroll_worker_thread, task);
    g_thread_unref(thread);
}

/* ==================== VERIFY ASYNC WORKER ==================== */

typedef struct {
    FpDevice *device;
    FpPrint *print;
    unsigned char *saved_data;
    int saved_size;
    int match_result;
} VerifyTaskData;

static gboolean elan_verify_complete_idle(gpointer user_data) {
    VerifyTaskData *task = (VerifyTaskData *)user_data;

    fpi_device_verify_report(task->device,
                             task->match_result ? FPI_MATCH_SUCCESS : FPI_MATCH_FAIL,
                             task->match_result ? task->print : NULL,
                             NULL);
    fpi_device_verify_complete(task->device, NULL);

    if (task->saved_data) g_free(task->saved_data);
    g_free(task);
    return G_SOURCE_REMOVE;
}

static gpointer elan_verify_worker_thread(gpointer user_data) {
    VerifyTaskData *task = (VerifyTaskData *)user_data;
    if (task->saved_data && task->saved_size > 0) {
        task->match_result = sift_engine_verify(task->saved_data, task->saved_size);
    } else {
        task->match_result = 0;
    }
    g_idle_add(elan_verify_complete_idle, task);
    return NULL;
}

static void elan_verify(FpDevice *device) {
    FpPrint *print = NULL;
    fpi_device_get_verify_data(device, &print);

    VerifyTaskData *task = g_new0(VerifyTaskData, 1);
    task->device = device;
    task->print = print;

    if (print) {
        GVariant *record = NULL;
        g_object_get(print, "fpi-data", &record, NULL);
        if (record) {
            gsize size = 0;
            const unsigned char *bytes = (const unsigned char *)g_variant_get_fixed_array(record, &size, sizeof(guchar));
            if (bytes && size > 0) {
                task->saved_data = (unsigned char *)g_memdup2(bytes, size);
                task->saved_size = (int)size;
            }
            g_variant_unref(record);
        }
    }

    GThread *thread = g_thread_new("elan-verify-worker", elan_verify_worker_thread, task);
    g_thread_unref(thread);
}

static const FpIdEntry elan_id_table[] = {
    { .vid = 0x04f3, .pid = 0x0c4f },
    { .vid = 0, .pid = 0 }
};

static void fpi_device_elan_class_init(FpiDeviceElanClass *klass) {
    FpDeviceClass *dev_class = FP_DEVICE_CLASS(klass);

    dev_class->id = "elan";
    dev_class->full_name = "Elan Trojan SIFT";
    dev_class->type = FP_DEVICE_TYPE_USB;
    dev_class->id_table = elan_id_table;
    dev_class->scan_type = FP_SCAN_TYPE_PRESS;
    dev_class->nr_enroll_stages = 1;
    dev_class->features = FP_DEVICE_FEATURE_VERIFY | FP_DEVICE_FEATURE_ENROLL;
    
    dev_class->open = elan_open;
    dev_class->close = elan_close;
    dev_class->enroll = elan_enroll;
    dev_class->verify = elan_verify;
}

static void fpi_device_elan_init(FpiDeviceElan *self) {
}