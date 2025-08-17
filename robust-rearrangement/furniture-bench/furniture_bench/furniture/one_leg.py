from furniture_bench.furniture.square_table import SquareTable


class OneLeg(SquareTable):
    def __init__(self):
        super().__init__()
        self.should_be_assembled = [(0, 4)]
    
    def randomize_init_pose(self, from_skill, pos_range=[-0.05, 0.05], rot_range=45) -> bool:
        """Override to disable randomization for one_leg - keep ALL parts at fixed positions"""
        # Keep ALL parts at their exact original positions - no randomization at all
        for i, part in enumerate(self.parts):
            # Reset to exact original position and orientation from config
            part.reset_pos[from_skill] = part.part_config["reset_pos"][from_skill].copy()
            part.mut_ori = part.part_config.get("mut_ori", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            part.reset_ori[from_skill] = part.part_config["reset_ori"][from_skill].copy()
        
        # Always return True since we're using fixed positions
        return True
