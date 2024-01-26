import bpy
import numpy as np
import cv2
import mediapipe as mp


class ModalLiveFaceCapture(bpy.types.Operator):
    """Operator which runs itself from a timer"""
    bl_idname = "wm.modal_timer_operator"
    bl_label = "Modal Timer Operator"

    _timer = None

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            # change theme color, silly!
            self.video_capture.grab()
            _, frame = self.video_capture.retrieve()
            
            detection_result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
            for i in len(detection_result.face_blendshapes[0]):
                name, value = detection_result.face_blendshapes[0][i].category_name, detection_result.face_blendshapes[0][i].score
                if name in self.object.data.shape_keys.key_blocks:
                    self.object.data.shape_keys.key_blocks[name].value = value
                else:
                    print(f'WARNING: {name} not found in {self.object.name}')

        return {'PASS_THROUGH'}

    def execute(self, context):
        base_options = mp.tasks.BaseOptions(model_asset_path=r'C:\Users\t-ruiwang\Documents\projects\live_mocap\mediapipe-assets\face_landmarker_v2_with_blendshapes.task')
        options = mp.tasks.vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               running_mode=mp.tasks.vision.RunningMode.IMAGE,
                                               num_faces=1)
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        self.video_capture = cv2.VideoCapture(0)

        self.object = bpy.context.active_object
        print(self.object.name)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)

        self.video_capture.release()
        self.detector.close()



def menu_func(self, context):
    self.layout.operator(ModalTimerOperator.bl_idname, text=ModalTimerOperator.bl_label)


def register():
    bpy.utils.register_class(ModalTimerOperator)
    bpy.types.VIEW3D_MT_view.append(menu_func)


# Register and add to the "view" menu (required to also use F3 search "Modal Timer Operator" for quick access).
def unregister():
    bpy.utils.unregister_class(ModalTimerOperator)
    bpy.types.VIEW3D_MT_view.remove(menu_func)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.wm.modal_timer_operator()
