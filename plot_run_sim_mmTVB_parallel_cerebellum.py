

## Import tools:
from tools_for_plot_mmTVB_parallel import *
import os
import argparse



def main():


    parser = argparse.ArgumentParser(description="\
                                    TVB multimodel.\
                                    Script for visualizing a simulation.\
                                    Usage: plot_run_sim_mmTVB_parallel SUB_ID date_of_res root_path prot_folder [options]\
                                     ")
    
    parser.add_argument("SUB_ID", help="Subject ID")
    parser.add_argument("date_of_res", help="Prefix of TVB output files is the date YYYYMMGG_HHMMSS")
    parser.add_argument("--prot_folder", help="Protocol folder", default='/home/bcc/HCP_TVBmm_30M')
    parser.add_argument("--tvb_out_dirname", help="TVB output folder", default='TVB_output')
    parser.add_argument("--show_plot", help="Show plot. NOT suggested for terminal programming", type=bool, default=False)
    parser.add_argument("--save_plot", help="Save fig in png in working directory", type = bool, default=True)


    args = parser.parse_args()
    dateofres = args.date_of_res

    prot_dirname = args.prot_folder
    sub_id = args.SUB_ID
    tvb_out_dirname = args.tvb_out_dirname

    sub_tvb_outfolder = os.path.join(prot_dirname, sub_id, tvb_out_dirname)
    
    #t_mf, mf_act_crbl, mf_act_cerebral, bold_t, bolddata = load_data(dateofres, sub_tvb_outfolder)

    t_mf, mf_act_crbl, bold_t, bolddata = load_data_only_crbl(dateofres, sub_tvb_outfolder)

    t_tr = int(0.03*t_mf[-1]) # transient = 5% t
    vol_tr = 50 # fixed to first 5 volumes   

    plot_TVB_crblMF_activity(sub_tvb_outfolder, dateofres, sub_id, t_mf, t_tr, mf_act_crbl, show_plot = args.show_plot, save_bool=args.save_plot)
    #plot_TVB_crblMF_activity_2pop(sub_tvb_outfolder, dateofres, sub_id, t_mf, t_tr, mf_act_crbl, show_plot = args.show_plot, save_bool=args.save_plot)

    plot_TVB_crblBOLD_only_cortex(sub_tvb_outfolder,dateofres, sub_id, bold_t, vol_tr, bolddata, show_plot = args.show_plot, save_bool=args.save_plot)
    #plot_TVB_crblBOLD(sub_tvb_outfolder,dateofres, sub_id, bold_t, vol_tr, bolddata, show_plot = args.show_plot, save_bool=args.save_plot)
    #plot_TVB_crblBOLD_2pop(sub_tvb_outfolder,dateofres, sub_id, bold_t, vol_tr, bolddata, show_plot = args.show_plot, save_bool=args.save_plot)

    #plot_TVB_cerebral_activity(sub_tvb_outfolder, dateofres, sub_id, t_mf, t_tr, mf_act_cerebral, show_plot = args.show_plot, save_bool=args.save_plot)
    #plot_TVB_cerebralBOLD(sub_tvb_outfolder,dateofres, sub_id, bold_t, vol_tr, bolddata, show_plot=args.show_plot, save_bool=args.save_plot)

if __name__ == "__main__":
    main()


