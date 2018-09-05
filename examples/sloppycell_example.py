from __future__ import print_function
import sys

from PyDSTool import *
from PyDSTool.Toolbox.makeSloppyModel import *
from time import clock

sloppyModelEg = {
 'assignments': {'CLB2T_21': 'CLB2_20 + C2_4 + C2P_5 + F2_29 + F2P_30',
                 'F_28': 'exp(-mu_39 * D_26)',
                 'mu_39': 'log(2) / mdt_216',
                 'Vacdh_58': 'kacdh_126 + kacdh_127 * CDC14_8',
                 'CDC14T_9': 'CDC14_8 + RENT_47 + RENTP_48',
                 'Vppc1_72': 'kppc1_183 * CDC14_8',
                 'D_26': '1.026 / mu_39 - 32',
                 'Vasbf_60': 'kasbf_132 * (esbfn2_101 * CLN2_24 + esbfn3_102 * (CLN3_25 + BCK2_1) + esbfb5_100 * CLB5_22)',
                 'Vd2c1_61': 'kd2c1_144 * (ec1n3_88 * CLN3_25 + ec1k2_86 * BCK2_1 + ec1n2_87 * CLN2_24 + ec1b5_85 * CLB5_22 + ec1b2_84 * CLB2_20)',
                 'Vppf6_73': 'kppf6_184 * CDC14_8',
                 'Vaiep_59': 'kaiep_128 * CLB2_20',
                 'CLN3_25': 'C0_78 * Dn3_80 * MASS_37 / (Jn3_118 + Dn3_80 * MASS_37)',
                 'CLB5T_23': 'CLB5_22 + C5_6 + C5P_7 + F5_31 + F5P_32',
                 'SBF_49': 'GK_219(Vasbf_60, Visbf_68, Jasbf_109, Jisbf_116)',
                 'SIC1T_52': 'SIC1_50 + C2_4 + C5_6 + SIC1P_51 + C2P_5 + C5P_7',
                 'Vdpds_65': 'kdpds_142 + kdpds_146 * CDC20_12 + kdpds_149 * CDH1_17',
                 'Vdb2_63': 'kdb_150 + kdb_151 * CDH1_17 + kdb2p_152 * CDC20_12',
                 'Vkpf6_70': 'kd1f6_141 + Vd2f6_62 / (Jd2f6_112 + CDC6_14 + F2_29 + F5_31 + CDC6P_15 + F2P_30 + F5P_32)',
                 'Vkpnet_71': '(kkpnet_181 + kkpnet_182 * CDC15_10) * MASS_37',
                 'NET1T_42': 'NET1_40 + NET1P_41 + RENT_47 + RENTP_48',
                 'Vd2f6_62': 'kd2f6_145 * (ef6n3_93 * CLN3_25 + ef6k2_91 * BCK2_1 + ef6n2_92 * CLN2_24 + ef6b5_90 * CLB5_22 + ef6b2_89 * CLB2_20)',
                 'BCK2_1': 'b0_75 * MASS_37',
                 'Visbf_68': 'kisbf_178 + kisbf_179 * CLB2_20',
                 'Vdb5_64': 'kdb_153 + kdb_154 * CDC20_12',
                 'Vicdh_67': 'kicdh_174 + kicdh_175 * (eicdhn3_97 * CLN3_25 + eicdhn2_96 * CLN2_24 + eicdhb5_95 * CLB5_22 + eicdhb2_94 * CLB2_20)',
                 'Vppnet_74': 'kppnet_185 + kppnet_186 * PPX_46',
                 'MCM1_38': 'GK_219(kamcm_129 * CLB2_20, kimcm_177, Jamcm_108, Jimcm_115)',
                 'CKIT_19': 'SIC1T_52 + CDC6T_16',
                 'Vdppx_66': 'kdppx_167 + kdppx_168 * (J20ppx_105 + CDC20_12) * Jpds_119 / (Jpds_119 + PDS1_44)',
                 'CDC6T_16': 'CDC6_14 + F2_29 + F5_31 + CDC6P_15 + F2P_30 + F5P_32',
                 'Vkpc1_69': 'kd1c1_140 + Vd2c1_61 / (Jd2c1_111 + SIC1_50 + C2_4 + C5_6 + SIC1P_51 + C2P_5 + C5P_7)'},
 'functions': {'MichaelisMenten_220(M1, J1, k1, S1)': 'k1 * S1 * M1 / (J1 + S1)',
               'Mass_Action_0_223(k1)': 'k1',
               'GK_219(A1, A2, A3, A4)': '2 * A4 * A1 / (BB_218(A1, A2, A3, A4) + sqrt(pow(BB_218(A1, A2, A3, A4), 2) - 4 * (A2 - A1) * A4 * A1))',
               'BB_218(A1, A2, A3, A4)': 'A2 - A1 + A3 * A2 + A4 * A1',
               'Mass_Action_1_222(k1, S1)': 'k1 * S1',
               'Mass_Action_2_221(k1, S1, S2)': 'k1 * S1 * S2'},
 'odes': {'IE_33': '- ( MichaelisMenten_220(Vaiep_59, Jaiep_107, 1, IE_33) ) + ( MichaelisMenten_220(1, Jiiep_114, kiiep_176, IEP_34) )', 'CDC20i_13': '( Mass_Action_0_223(ks_189 + ks_190 * MCM1_38) ) - ( Mass_Action_1_222(kd20_143, CDC20i_13) ) - ( Mass_Action_1_222(ka_124 + ka_125 * IEP_34, CDC20i_13) ) + ( Mass_Action_1_222(MAD2_36, CDC20_12) )', 'PE_45': '- ( Mass_Action_1_222(Vdpds_65, PE_45) ) + ( Mass_Action_2_221(kasesp_133, PDS1_44, ESP1_27) ) - ( Mass_Action_1_222(kdiesp_159, PE_45) )', 'CDH1i_18': '- ( Mass_Action_1_222(kdcdh_156, CDH1i_18) ) - ( MichaelisMenten_220(Vacdh_58, Jacdh_106, 1, CDH1i_18) ) + ( MichaelisMenten_220(Vicdh_67, Jicdh_113, 1, CDH1_17) )', 'PDS1_44': '( Mass_Action_0_223(kspds_207 + kspds_188 * SBF_49 + kspds_191 * MCM1_38) ) - ( Mass_Action_1_222(Vdpds_65, PDS1_44) ) - ( Mass_Action_2_221(kasesp_133, PDS1_44, ESP1_27) ) + ( Mass_Action_1_222(kdiesp_159, PE_45) )', 'CDC15i_11': '- ( Mass_Action_1_222(ka_121 * TEM1GDP_56 + ka_122 * TEM1GTP_57 + ka15p_123 * CDC14_8, CDC15i_11) ) + ( Mass_Action_1_222(ki15_173, CDC15_10) )', 'F5_31': '( Mass_Action_2_221(kasf5_135, CLB5_22, CDC6_14) ) - ( Mass_Action_1_222(kdif5_161, F5_31) ) - ( Mass_Action_1_222(Vkpf6_70, F5_31) ) + ( Mass_Action_1_222(Vppf6_73, F5P_32) ) - ( Mass_Action_1_222(Vdb5_64, F5_31) )', 'ESP1_27': '( Mass_Action_1_222(Vdpds_65, PE_45) ) - ( Mass_Action_2_221(kasesp_133, PDS1_44, ESP1_27) ) + ( Mass_Action_1_222(kdiesp_159, PE_45) )', 'SIC1_50': '( Mass_Action_0_223(ksc_197 + ksc_198 * SWI5_54) ) - ( Mass_Action_1_222(Vkpc1_69, SIC1_50) ) + ( Mass_Action_1_222(Vppc1_72, SIC1P_51) ) - ( Mass_Action_2_221(kasb2_130, CLB2_20, SIC1_50) ) + ( Mass_Action_1_222(kdib2_157, C2_4) ) - ( Mass_Action_2_221(kasb5_131, CLB5_22, SIC1_50) ) + ( Mass_Action_1_222(kdib5_158, C5_6) ) + ( Mass_Action_1_222(Vdb2_63, C2_4) ) + ( Mass_Action_1_222(Vdb5_64, C5_6) )', 'CDH1_17': '( Mass_Action_0_223(kscdh_199) ) - ( Mass_Action_1_222(kdcdh_156, CDH1_17) ) + ( MichaelisMenten_220(Vacdh_58, Jacdh_106, 1, CDH1i_18) ) - ( MichaelisMenten_220(Vicdh_67, Jicdh_113, 1, CDH1_17) )', 'TEM1GDP_56': '- ( MichaelisMenten_220(LTE1_35, Jatem_110, 1, TEM1GDP_56) ) + ( MichaelisMenten_220(BUB2_2, Jitem_117, 1, TEM1GTP_57) )', 'LTE1_35': '( Mass_Action_0_223(0) )', 'NET1P_41': '- ( Mass_Action_2_221(kasrentp_137, CDC14_8, NET1P_41) ) + ( Mass_Action_1_222(kdirentp_163, RENTP_48) ) - ( Mass_Action_1_222(kdnet_165, NET1P_41) ) + ( Mass_Action_1_222(Vkpnet_71, NET1_40) ) - ( Mass_Action_1_222(Vppnet_74, NET1P_41) ) + ( Mass_Action_1_222(kd14_139, RENTP_48) )', 'C5P_7': '( Mass_Action_1_222(Vkpc1_69, C5_6) ) - ( Mass_Action_1_222(Vppc1_72, C5P_7) ) - ( Mass_Action_1_222(kd3c1_147, C5P_7) ) - ( Mass_Action_1_222(Vdb5_64, C5P_7) )', 'F2P_30': '( Mass_Action_1_222(Vkpf6_70, F2_29) ) - ( Mass_Action_1_222(Vppf6_73, F2P_30) ) - ( Mass_Action_1_222(kd3f6_148, F2P_30) ) - ( Mass_Action_1_222(Vdb2_63, F2P_30) )', 'RENTP_48': '( Mass_Action_2_221(kasrentp_137, CDC14_8, NET1P_41) ) - ( Mass_Action_1_222(kdirentp_163, RENTP_48) ) + ( Mass_Action_1_222(Vkpnet_71, RENT_47) ) - ( Mass_Action_1_222(Vppnet_74, RENTP_48) ) - ( Mass_Action_1_222(kdnet_165, RENTP_48) ) - ( Mass_Action_1_222(kd14_139, RENTP_48) )', 'CLB5_22': '( Mass_Action_0_223((ksb_194 + ksb_195 * SBF_49) * MASS_37) ) - ( Mass_Action_1_222(Vdb5_64, CLB5_22) ) - ( Mass_Action_2_221(kasb5_131, CLB5_22, SIC1_50) ) + ( Mass_Action_1_222(kdib5_158, C5_6) ) + ( Mass_Action_1_222(kd3c1_147, C5P_7) ) - ( Mass_Action_2_221(kasf5_135, CLB5_22, CDC6_14) ) + ( Mass_Action_1_222(kdif5_161, F5_31) ) + ( Mass_Action_1_222(kd3f6_148, F5P_32) )', 'F5P_32': '( Mass_Action_1_222(Vkpf6_70, F5_31) ) - ( Mass_Action_1_222(Vppf6_73, F5P_32) ) - ( Mass_Action_1_222(kd3f6_148, F5P_32) ) - ( Mass_Action_1_222(Vdb5_64, F5P_32) )', 'NET1_40': '- ( Mass_Action_2_221(kasrent_136, CDC14_8, NET1_40) ) + ( Mass_Action_1_222(kdirent_162, RENT_47) ) + ( Mass_Action_0_223(ksnet_205) ) - ( Mass_Action_1_222(kdnet_165, NET1_40) ) - ( Mass_Action_1_222(Vkpnet_71, NET1_40) ) + ( Mass_Action_1_222(Vppnet_74, NET1P_41) ) + ( Mass_Action_1_222(kd14_139, RENT_47) )', 'C5_6': '( Mass_Action_2_221(kasb5_131, CLB5_22, SIC1_50) ) - ( Mass_Action_1_222(kdib5_158, C5_6) ) - ( Mass_Action_1_222(Vkpc1_69, C5_6) ) + ( Mass_Action_1_222(Vppc1_72, C5P_7) ) - ( Mass_Action_1_222(Vdb5_64, C5_6) )', 'CDC15_10': '( Mass_Action_1_222(ka_121 * TEM1GDP_56 + ka_122 * TEM1GTP_57 + ka15p_123 * CDC14_8, CDC15i_11) ) - ( Mass_Action_1_222(ki15_173, CDC15_10) )', 'C2_4': '( Mass_Action_2_221(kasb2_130, CLB2_20, SIC1_50) ) - ( Mass_Action_1_222(kdib2_157, C2_4) ) - ( Mass_Action_1_222(Vkpc1_69, C2_4) ) + ( Mass_Action_1_222(Vppc1_72, C2P_5) ) - ( Mass_Action_1_222(Vdb2_63, C2_4) )', 'TEM1GTP_57': '( MichaelisMenten_220(LTE1_35, Jatem_110, 1, TEM1GDP_56) ) - ( MichaelisMenten_220(BUB2_2, Jitem_117, 1, TEM1GTP_57) )', 'CLB2_20': '( Mass_Action_0_223((ksb_192 + ksb_193 * MCM1_38) * MASS_37) ) - ( Mass_Action_1_222(Vdb2_63, CLB2_20) ) - ( Mass_Action_2_221(kasb2_130, CLB2_20, SIC1_50) ) + ( Mass_Action_1_222(kdib2_157, C2_4) ) + ( Mass_Action_1_222(kd3c1_147, C2P_5) ) - ( Mass_Action_2_221(kasf2_134, CLB2_20, CDC6_14) ) + ( Mass_Action_1_222(kdif2_160, F2_29) ) + ( Mass_Action_1_222(kd3f6_148, F2P_30) )', 'PPX_46': '( Mass_Action_0_223(ksppx_208) ) - ( Mass_Action_1_222(Vdppx_66, PPX_46) )', 'C2P_5': '( Mass_Action_1_222(Vkpc1_69, C2_4) ) - ( Mass_Action_1_222(Vppc1_72, C2P_5) ) - ( Mass_Action_1_222(kd3c1_147, C2P_5) ) - ( Mass_Action_1_222(Vdb2_63, C2P_5) )', 'BUB2_2': '( Mass_Action_0_223(0) )', 'CDC6P_15': '( Mass_Action_1_222(Vkpf6_70, CDC6_14) ) - ( Mass_Action_1_222(Vppf6_73, CDC6P_15) ) - ( Mass_Action_1_222(kd3f6_148, CDC6P_15) ) + ( Mass_Action_1_222(Vdb2_63, F2P_30) ) + ( Mass_Action_1_222(Vdb5_64, F5P_32) )', 'MAD2_36': '( Mass_Action_0_223(0) )', 'CDC20_12': '- ( Mass_Action_1_222(kd20_143, CDC20_12) ) + ( Mass_Action_1_222(ka_124 + ka_125 * IEP_34, CDC20i_13) ) - ( Mass_Action_1_222(MAD2_36, CDC20_12) )', 'BUD_3': '( Mass_Action_0_223(ksbud_196 * (ebudn2_82 * CLN2_24 + ebudn3_83 * CLN3_25 + ebudb5_81 * CLB5_22)) ) - ( Mass_Action_1_222(kdbud_155, BUD_3) )', 'F2_29': '( Mass_Action_2_221(kasf2_134, CLB2_20, CDC6_14) ) - ( Mass_Action_1_222(kdif2_160, F2_29) ) - ( Mass_Action_1_222(Vkpf6_70, F2_29) ) + ( Mass_Action_1_222(Vppf6_73, F2P_30) ) - ( Mass_Action_1_222(Vdb2_63, F2_29) )', 'SIC1P_51': '( Mass_Action_1_222(Vkpc1_69, SIC1_50) ) - ( Mass_Action_1_222(Vppc1_72, SIC1P_51) ) - ( Mass_Action_1_222(kd3c1_147, SIC1P_51) ) + ( Mass_Action_1_222(Vdb2_63, C2P_5) ) + ( Mass_Action_1_222(Vdb5_64, C5P_7) )', 'CDC14_8': '( Mass_Action_0_223(ks14_187) ) - ( Mass_Action_1_222(kd14_139, CDC14_8) ) - ( Mass_Action_2_221(kasrent_136, CDC14_8, NET1_40) ) + ( Mass_Action_1_222(kdirent_162, RENT_47) ) - ( Mass_Action_2_221(kasrentp_137, CDC14_8, NET1P_41) ) + ( Mass_Action_1_222(kdirentp_163, RENTP_48) ) + ( Mass_Action_1_222(kdnet_165, RENT_47) ) + ( Mass_Action_1_222(kdnet_165, RENTP_48) )', 'RENT_47': '( Mass_Action_2_221(kasrent_136, CDC14_8, NET1_40) ) - ( Mass_Action_1_222(kdirent_162, RENT_47) ) - ( Mass_Action_1_222(Vkpnet_71, RENT_47) ) + ( Mass_Action_1_222(Vppnet_74, RENTP_48) ) - ( Mass_Action_1_222(kdnet_165, RENT_47) ) - ( Mass_Action_1_222(kd14_139, RENT_47) )', 'IEP_34': '( MichaelisMenten_220(Vaiep_59, Jaiep_107, 1, IE_33) ) - ( MichaelisMenten_220(1, Jiiep_114, kiiep_176, IEP_34) )', 'MASS_37': '( Mass_Action_0_223(mu_39 * MASS_37) )', 'SWI5_54': '( Mass_Action_0_223(ksswi_210 + ksswi_211 * MCM1_38) ) - ( Mass_Action_1_222(kdswi_170, SWI5_54) ) + ( Mass_Action_1_222(kaswi_138 * CDC14_8, SWI5P_55) ) - ( Mass_Action_1_222(kiswi_180 * CLB2_20, SWI5_54) )', 'CLN2_24': '( Mass_Action_0_223((ksn_203 + ksn_204 * SBF_49) * MASS_37) ) - ( Mass_Action_1_222(kdn2_164, CLN2_24) )', 'ORI_43': '( Mass_Action_0_223(ksori_206 * (eorib5_99 * CLB5_22 + eorib2_98 * CLB2_20)) ) - ( Mass_Action_1_222(kdori_166, ORI_43) )', 'CDC6_14': '( Mass_Action_0_223(ksf_200 + ksf_201 * SWI5_54 + ksf_202 * SBF_49) ) - ( Mass_Action_1_222(Vkpf6_70, CDC6_14) ) + ( Mass_Action_1_222(Vppf6_73, CDC6P_15) ) - ( Mass_Action_2_221(kasf2_134, CLB2_20, CDC6_14) ) + ( Mass_Action_1_222(kdif2_160, F2_29) ) - ( Mass_Action_2_221(kasf5_135, CLB5_22, CDC6_14) ) + ( Mass_Action_1_222(kdif5_161, F5_31) ) + ( Mass_Action_1_222(Vdb2_63, F2_29) ) + ( Mass_Action_1_222(Vdb5_64, F5_31) )',
          'SWI5P_55': '- ( Mass_Action_1_222(kdswi_170, SWI5P_55) ) - ( Mass_Action_1_222(kaswi_138 * CDC14_8, SWI5P_55) ) + ( Mass_Action_1_222(kiswi_180 * CLB2_20, SWI5_54) )',
          'SPN_53': '( Mass_Action_0_223(ksspn_209 * CLB2_20 / (Jspn_120 + CLB2_20)) ) - ( Mass_Action_1_222(kdspn_169, SPN_53) )'},
 'parameters': {'kd3f6_148': 1.0, 'kacdh_126': 0.01, 'kacdh_127': 0.80000000000000004, 'KEZ_171': 0.29999999999999999,
                'kdirent_162': 1.0, 'Jatem_110': 0.10000000000000001, 'kspds_188': 0.029999999999999999,
                'ef6n2_92': 0.059999999999999998, 'ef6b5_90': 0.10000000000000001, 'ks_189': 0.0060000000000000001,
                'kimcm_177': 0.14999999999999999, 'ksc_197': 0.012, 'ebudn2_82': 0.25, 'eicdhb2_94': 1.2,
                'Jacdh_106': 0.029999999999999999, 'ksspn_209': 0.10000000000000001, 'eicdhn2_96': 0.40000000000000002,
                'kamcm_129': 1.0, 'kdiesp_159': 0.5, 'ef6k2_91': 0.029999999999999999, 'Jamcm_108': 0.10000000000000001,
                'kdif5_161': 0.01, 'mad2h_214': 8.0, 'ks_190': 0.59999999999999998, 'kiiep_176': 0.14999999999999999,
                'kscdh_199': 0.01, 'ksn_203': 0.0, 'ebudn3_83': 0.050000000000000003, 'ka_124': 0.050000000000000003,
                'ka_125': 0.20000000000000001, 'TEM1T_217': 1.0, 'ka_121': 0.002, 'ka_122': 1.0, 'kdbud_155': 0.059999999999999998,
                'Dn3_80': 1.0, 'kdpds_142': 0.01, 'ec1b5_85': 0.10000000000000001, 'kdpds_146': 0.20000000000000001,
                'kdpds_149': 0.040000000000000001, 'ksn_204': 0.14999999999999999, 'kd3c1_147': 1.0,
                'ec1k2_86': 0.029999999999999999, 'Jicdh_113': 0.029999999999999999, 'Jitem_117': 0.10000000000000001,
                'lte1h_212': 1.0, 'kasb5_131': 50.0, 'ec1n3_88': 0.29999999999999999, 'ebudb5_81': 1.0,
                'esbfb5_100': 2.0, 'Jn3_118': 6.0, 'kasbf_132': 0.38, 'KEZ2_172': 0.20000000000000001,
                'kdspn_169': 0.059999999999999998, 'kasrent_136': 200.0, 'kppc1_183': 4.0, 'cell_0': 1.0,
                'mad2l_215': 0.01, 'ksori_206': 2.0, 'Jpds_119': 0.040000000000000001, 'kaswi_138': 2.0,
                'kkpnet_181': 0.01, 'kkpnet_182': 0.59999999999999998, 'kdori_166': 0.059999999999999998,
                'kasb2_130': 50.0, 'kdppx_168': 2.0, 'kasesp_133': 50.0, 'ksbud_196': 0.20000000000000001,
                'kdb2p_152': 0.14999999999999999, 'Jspn_120': 0.14000000000000001, 'kppnet_186': 3.0, 'ksc_198': 0.12,
                'kisbf_178': 0.59999999999999998, 'kdirentp_163': 2.0, 'Jd2c1_111': 0.050000000000000003,
                'eorib5_99': 0.90000000000000002, 'ec1b2_84': 0.45000000000000001, 'kd2c1_144': 1.0,
                'ksnet_205': 0.084000000000000005, 'IET_104': 1.0, 'esbfn3_102': 10.0, 'kd2f6_145': 1.0,
                'kd14_139': 0.10000000000000001, 'ec1n2_87': 0.059999999999999998, 'Jaiep_107': 0.10000000000000001,
                'kicdh_175': 0.080000000000000002, 'kicdh_174': 0.001, 'Jasbf_109': 0.01, 'ef6b2_89': 0.55000000000000004,
                'esbfn2_101': 2.0, 'kdb_153': 0.01, 'eorib2_98': 0.45000000000000001, 'ks14_187': 0.20000000000000001,
                'kdnet_165': 0.029999999999999999, 'kasrentp_137': 1.0, 'kdppx_167': 0.17000000000000001, 'ki15_173': 0.5,
                'lte1l_213': 0.10000000000000001, 'kspds_207': 0.0, 'ksppx_208': 0.10000000000000001, 'kd1f6_141': 0.01,
                'C0_78': 0.40000000000000002, 'kspds_191': 0.055, 'ef6n3_93': 0.29999999999999999, 'kasf5_135': 0.01,
                'Jd2f6_112': 0.050000000000000003, 'kppf6_184': 4.0, 'kdb_154': 0.16, 'kdb_151': 0.40000000000000002,
                'kdb_150': 0.0030000000000000001, 'ka15p_123': 0.001, 'kdib2_157': 0.050000000000000003,
                'ksswi_210': 0.0050000000000000001, 'ksswi_211': 0.080000000000000002, 'kiswi_180': 0.050000000000000003,
                'kppnet_185': 0.050000000000000003, 'eicdhn3_97': 0.25, 'ESP1T_103': 1.0, 'Jisbf_116': 0.01,
                'J20ppx_105': 0.14999999999999999, 'kdif2_160': 0.5, 'kd20_143': 0.29999999999999999, 'CDC15T_79': 1.0,
                'Jiiep_114': 0.10000000000000001, 'ksb_194': 0.00080000000000000004, 'eicdhb5_95': 8.0, 'ksb_192': 0.001,
                'ksb_193': 0.040000000000000001, 'ksb_195': 0.0050000000000000001, 'kasf2_134': 15.0,
                'b0_75': 0.053999999999999999, 'kdcdh_156': 0.01, 'kisbf_179': 8.0, 'bub2h_76': 1.0, 'mdt_216': 90.0,
                'kaiep_128': 0.10000000000000001, 'kdswi_170': 0.080000000000000002, 'kdn2_164': 0.12,
                'bub2l_77': 0.20000000000000001, 'kd1c1_140': 0.01, 'ksf_200': 0.024, 'ksf_201': 0.12,
                'ksf_202': 0.0040000000000000001, 'kdib5_158': 0.059999999999999998, 'Jimcm_115': 0.10000000000000001},
 'events': {'lt(CLB2_20 - KEZ_171, 0)': {'MASS_37': 'F_28 * MASS_37', 'SPN_53': '0', 'BUD_3': '0', 'LTE1_35': 'lte1l_213'},
            'gt(ORI_43 - 1, 0)': {'MAD2_36': 'mad2h_214', 'BUB2_2': 'bub2h_76'},
            'lt(CLB2_20 + CLB5_22 - KEZ2_172, 0)': {'ORI_43': '0'},
            'gt(SPN_53 - 1, 0)': {'MAD2_36': 'mad2l_215', 'BUB2_2': 'bub2l_77', 'LTE1_35': 'lte1h_212'}},
 'domains': {'kd1c1_140': [0,0.5], 'SPN_53': [-100,100]}} # example: one parameter and one variable

all_ics_pars = {'BCK2_1': 0.0,
 'BUB2_2': 0.20000000000000001,
 'BUD_3': 0.0084729999999999996,
 'C0_78': 0.40000000000000002,
 'C2P_5': 0.024034,
 'C2_4': 0.238404,
 'C5P_7': 0.0068780000000000004,
 'C5_6': 0.070081000000000004,
 'CDC14T_9': 2.0,
 'CDC14_8': 0.46834399999999998,
 'CDC15T_79': 1.0,
 'CDC15_10': 0.65653300000000003,
 'CDC15i_11': 0.34346599999999999,
 'CDC20_12': 0.44429600000000002,
 'CDC20i_13': 1.4720439999999999,
 'CDC6P_15': 0.015486,
 'CDC6T_16': 0.0,
 'CDC6_14': 0.10758,
 'CDH1_17': 0.93049899999999997,
 'CDH1i_18': 0.069500000000000006,
 'CKIT_19': 0.0,
 'CLB2T_21': 0.17000000000000001,
 'CLB2_20': 0.14692269999999999,
 'CLB5T_23': 0.12,
 'CLB5_22': 0.051801399999999997,
 'CLN2_24': 0.065251100000000006,
 'CLN3_25': 0.0,
 'D_26': 101.2184600757, #0.0,
 'Dn3_80': 1.0,
 'ESP1T_103': 1.0,
 'ESP1_27': 0.301313,
 'F2P_30': 0.027393799999999999,
 'F2_29': 0.23605799999999999,
 'F5P_32': 7.9099999999999998e-06, # paper says e-06, Ryan says e-05
 'F5_31': 7.2399999999999998e-06, # e-06 ?
 'F_28': 0.458613409396, # was 0.0, but is supposed to = exp(-mu_39 * D_26)
 'IEP_34': 0.10150000000000001,
 'IET_104': 1.0,
 'IE_33': 0.89849999999999997,
 'J20ppx_105': 0.14999999999999999,
 'Jacdh_106': 0.029999999999999999,
 'Jaiep_107': 0.10000000000000001,
 'Jamcm_108': 0.10000000000000001,
 'Jasbf_109': 0.01,
 'Jatem_110': 0.10000000000000001,
 'Jd2c1_111': 0.050000000000000003,
 'Jd2f6_112': 0.050000000000000003,
 'Jicdh_113': 0.029999999999999999,
 'Jiiep_114': 0.10000000000000001,
 'Jimcm_115': 0.10000000000000001,
 'Jisbf_116': 0.01,
 'Jitem_117': 0.10000000000000001,
 'Jn3_118': 6.0,
 'Jpds_119': 0.040000000000000001,
 'Jspn_120': 0.14000000000000001,
 'KEZ2_172': 0.20000000000000001,
 'KEZ_171': 0.29999999999999999,
 'LTE1_35': 0.10000000000000001,
 'MAD2_36': 0.01,
 'MASS_37': 1.206019,
 'MCM1_38': 0.0,
 'NET1P_41': 0.97027099999999999,
 'NET1T_42': 2.7999999999999998,
 'NET1_40': 0.018644999999999998,
 'ORI_43': 0.00090899999999999998,
 'PDS1_44': 0.025611999999999999,
 'PE_45': 0.69999999999999996,
 'PPX_46': 0.123179,
 'RENTP_48': 0.59999999999999998,
 'RENT_47': 1.0495399999999999,
 'SBF_49': 0.0,
 'SIC1P_51': 0.0064099999999999999,
 'SIC1T_52': 0.0,
 'SIC1_50': 0.022877600000000001,
 'SPN_53': 0.029999999999999999,
 'SWI5P_55': 0.02,
 'SWI5_54': 0.94999999999999996,
 'TEM1GDP_56': 0.10000000000000001,
 'TEM1GTP_57': 0.90000000000000002,
 'TEM1T_217': 1.0,
 'Vacdh_58': 0.0,
 'Vaiep_59': 0.0,
 'Vasbf_60': 0.0,
 'Vd2c1_61': 0.0,
 'Vd2f6_62': 0.0,
 'Vdb2_63': 0.0,
 'Vdb5_64': 0.0,
 'Vdpds_65': 0.0,
 'Vdppx_66': 0.0,
 'Vicdh_67': 0.0,
 'Visbf_68': 0.0,
 'Vkpc1_69': 0.0,
 'Vkpf6_70': 0.0,
 'Vkpnet_71': 0.0,
 'Vppc1_72': 0.0,
 'Vppf6_73': 0.0,
 'Vppnet_74': 0.0,
 'b0_75': 0.053999999999999999,
 'bub2h_76': 1.0,
 'bub2l_77': 0.20000000000000001,
 'cell_0': 1.0,
 'ebudb5_81': 1.0,
 'ebudn2_82': 0.25,
 'ebudn3_83': 0.050000000000000003,
 'ec1b2_84': 0.45000000000000001,
 'ec1b5_85': 0.10000000000000001,
 'ec1k2_86': 0.029999999999999999,
 'ec1n2_87': 0.059999999999999998,
 'ec1n3_88': 0.29999999999999999,
 'ef6b2_89': 0.55000000000000004,
 'ef6b5_90': 0.10000000000000001,
 'ef6k2_91': 0.029999999999999999,
 'ef6n2_92': 0.059999999999999998,
 'ef6n3_93': 0.29999999999999999,
 'eicdhb2_94': 1.2,
 'eicdhb5_95': 8.0,
 'eicdhn2_96': 0.40000000000000002,
 'eicdhn3_97': 0.25,
 'eorib2_98': 0.45000000000000001,
 'eorib5_99': 0.90000000000000002,
 'esbfb5_100': 2.0,
 'esbfn2_101': 2.0,
 'esbfn3_102': 10.0,
 'ka15p_123': 0.001,
 'ka_121': 0.002,
 'ka_122': 1.0,
 'ka_124': 0.050000000000000003,
 'ka_125': 0.20000000000000001,
 'kacdh_126': 0.01,
 'kacdh_127': 0.80000000000000004,
 'kaiep_128': 0.10000000000000001,
 'kamcm_129': 1.0,
 'kasb2_130': 50.0,
 'kasb5_131': 50.0,
 'kasbf_132': 0.38,
 'kasesp_133': 50.0,
 'kasf2_134': 15.0,
 'kasf5_135': 0.01,
 'kasrent_136': 200.0,
 'kasrentp_137': 1.0,
 'kaswi_138': 2.0,
 'kd14_139': 0.10000000000000001,
 'kd1c1_140': 0.01,
 'kd1f6_141': 0.01,
 'kd20_143': 0.29999999999999999,
 'kd2c1_144': 1.0,
 'kd2f6_145': 1.0,
 'kd3c1_147': 1.0,
 'kd3f6_148': 1.0,
 'kdb2p_152': 0.14999999999999999,
 'kdb_150': 0.0030000000000000001,
 'kdb_151': 0.40000000000000002,
 'kdb_153': 0.01,
 'kdb_154': 0.16,
 'kdbud_155': 0.059999999999999998,
 'kdcdh_156': 0.01,
 'kdib2_157': 0.050000000000000003,
 'kdib5_158': 0.059999999999999998,
 'kdiesp_159': 0.5,
 'kdif2_160': 0.5,
 'kdif5_161': 0.01,
 'kdirent_162': 1.0,
 'kdirentp_163': 2.0,
 'kdn2_164': 0.12,
 'kdnet_165': 0.029999999999999999,
 'kdori_166': 0.059999999999999998,
 'kdpds_142': 0.01,
 'kdpds_146': 0.20000000000000001,
 'kdpds_149': 0.040000000000000001,
 'kdppx_167': 0.17000000000000001,
 'kdppx_168': 2.0,
 'kdspn_169': 0.059999999999999998,
 'kdswi_170': 0.080000000000000002,
 'ki15_173': 0.5,
 'kicdh_174': 0.001,
 'kicdh_175': 0.080000000000000002,
 'kiiep_176': 0.14999999999999999,
 'kimcm_177': 0.14999999999999999,
 'kisbf_178': 0.59999999999999998,
 'kisbf_179': 8.0,
 'kiswi_180': 0.050000000000000003,
 'kkpnet_181': 0.01,
 'kkpnet_182': 0.59999999999999998,
 'kppc1_183': 4.0,
 'kppf6_184': 4.0,
 'kppnet_185': 0.050000000000000003,
 'kppnet_186': 3.0,
 'ks14_187': 0.20000000000000001,
 'ks_189': 0.0060000000000000001,
 'ks_190': 0.59999999999999998,
 'ksb_192': 0.001,
 'ksb_193': 0.040000000000000001,
 'ksb_194': 0.00080000000000000004,
 'ksb_195': 0.0050000000000000001,
 'ksbud_196': 0.20000000000000001,
 'ksc_197': 0.012,
 'ksc_198': 0.12,
 'kscdh_199': 0.01,
 'ksf_200': 0.024,
 'ksf_201': 0.12,
 'ksf_202': 0.0040000000000000001,
 'ksn_203': 0.0,
 'ksn_204': 0.14999999999999999,
 'ksnet_205': 0.084000000000000005,
 'ksori_206': 2.0,
 'kspds_188': 0.029999999999999999,
 'kspds_191': 0.055,
 'kspds_207': 0.0,
 'ksppx_208': 0.10000000000000001,
 'ksspn_209': 0.10000000000000001,
 'ksswi_210': 0.0050000000000000001,
 'ksswi_211': 0.080000000000000002,
 'lte1h_212': 1.0,
 'lte1l_213': 0.10000000000000001,
 'mad2h_214': 8.0,
 'mad2l_215': 0.01,
 'mdt_216': 90.0,
 'mu_39': 0.007701635339555 # was 0.
                }

#print "F_28 should be ", exp(-all_ics_pars['mu_39']*all_ics_pars['D_26'])


# change genTarget to 'Vode_ODEsystem' if your external C compiler doesn't work

print("Making model")
genTarget = 'Radau_ODEsystem'
algparams = {'init_step': 0.1}

if genTarget == 'Vode_ODEsystem':
    algparams['stiff'] = True
sModel = makeSloppyModel('cplx_eg', sloppyModelEg, genTarget,
                     algParams=algparams, silent=True)

print("\nAux spec:\n", end=' ')
sModel.showDef('cplx_eg', 'auxspec')

# make some random i.c.s
def uniformICs(varnames, a, b):
    return dict(list(zip(varnames, [random.uniform(a, b) for v in varnames])))

#ics = uniformICs(sModel.allvars, 0, 1)

ics = {}
for name, val in all_ics_pars.items():
    if name in sModel.allvars:
        ics[name] = val


def compute(trajname='fig2', thi=205, dt=0.1, verboselevel=0):
    print("Computing trajectory")
    sModel.set(algparams={'init_step': dt})
    t0=clock()
    sModel.compute(trajname=trajname,
                   force=True,
                   ics=ics,
                   tdata=[0,thi],  # time in minutes
                   verboselevel=verboselevel
                  )
    print("Finished in %f seconds using initial step size of %f"%(clock()-t0,dt))


def doPlots(trajname='test', coords=None, dt=0.1, tlo=None, thi=None):
    plotdata = sModel.sample(trajname, coords=coords, dt=dt, tlo=tlo, thi=thi)
    f=plt.figure()
    for v in coords:
        plot(plotdata['t'], plotdata[v])

print("\nComputing trajectory")
compute()

# these plots correspond to the sub-plots of Chen 2004 paper
doPlots('fig2', ['MASS_37', 'SPN_53', 'ORI_43', 'BUD_3'])
doPlots('fig2', ['CLB2T_21', 'CLB5T_23', 'PDS1_44'])
doPlots('fig2', ['CDC6T_16', 'SIC1T_52', 'CDH1_17'])
doPlots('fig2', ['CLN2_24', 'CDC14_8', 'CDC20_12'])
doPlots('fig2', ['SBF_49', 'SWI5_54', 'MCM1_38'])

# Event0: v['CLB2_20']-p['KEZ_171']
# Event1: v['ORI_43']-1
# Event2: v['CLB2_20']+v['CLB5_22']-p['KEZ2_172']
# Event3: v['SPN_53']-1
evs=sModel.getTrajEvents('fig2')
print(orderEventData(sModel.getTrajEventTimes('fig2')))

show()
